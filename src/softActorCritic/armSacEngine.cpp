#include <torch/torch.h>
#include <algorithm>
#include <numeric>

#include "armSacEngine.h"


double ArmSacEngine::runOneEpisode(uint16_t seed, Learn::LearningMode mode){

    torch::Tensor newState;
    torch::Tensor actionTensor;
    float mulAction;
    uint64_t actionTaken;
    double singleReward;
    double result=0;
        

    armLearnEnv->reset(seed, mode);

    bool terminated = false;
    torch::Tensor state = getTensorState();

    uint64_t nbActions = 0;
    while (!terminated) {

        // Get the continuous action and the discretised one
        actionTensor = learningAgent.chooseAction(state);
        /*
        mulAction = (actionTensor.item<float>() + 1) * 4;
        actionTaken = 0;
        
        while(actionTaken + 1 < mulAction){
            actionTaken++;
        }

        armLearnEnv.doAction(actionTaken);*/

        std::vector<float> actionVector(actionTensor.data_ptr<float>(), actionTensor.data_ptr<float>() + actionTensor.numel());

        armLearnEnv->doActionContinuous(actionVector);
        newState = getTensorState();
        
        singleReward = armLearnEnv->getReward();
        
        nbActions++;
        terminated = (armLearnEnv->isTerminal() || nbActions == maxNbActions);

        if(mode == Learn::LearningMode::TRAINING){

            auto time = std::make_shared<std::chrono::time_point<
            std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());

            if(nbActions < maxNbActions)
                learningAgent.remember(state, actionTensor, singleReward, newState, terminated);

            learningAgent.learn();

            learningTime += ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *time)).count();


        }


        state = newState;
        result += singleReward;
    }
    
    return result;
}

void ArmSacEngine::trainOneGeneration(uint16_t nbIterationTraining){

    logNewGeneration();

    double result=0.0;
    double score=0.0;
    
    learningTime = 0.0;

    // we train the TPG, see doaction for the reward function
    for(int j = 0; j < nbIterationTraining; j++){

        uint64_t seed = generation * 100000 + j;
        result += runOneEpisode(seed, Learn::LearningMode::TRAINING);
        score += armLearnEnv->getScore();
    }
    result /= nbIterationTraining;
    score /= nbIterationTraining;
    logTraining(score, result);

    generation++;
    
}


void ArmSacEngine::validateOneGeneration(uint16_t nbIterationValidation){

    double score;
    // we train the TPG, see doaction for the reward function
    for(int j = 0; j < nbIterationValidation; j++){

        uint64_t seed = generation * 43000000000 + j;
        runOneEpisode(seed, Learn::LearningMode::VALIDATION);
        score += armLearnEnv->getScore();
    }
    score /= nbIterationValidation;

    logValidation(score);
};

void ArmSacEngine::validateTrainingOneGeneration(uint16_t nbIterationTrainingValidation){

    double score;
    // we train the TPG, see doaction for the reward function
    for(int j = 0; j < nbIterationTrainingValidation; j++){

        uint64_t seed = generation * 5000000 + j;
        runOneEpisode(seed, Learn::LearningMode::TESTING);
        score += armLearnEnv->getScore();
    }
    score /= nbIterationTrainingValidation;

    logTrainingValidation(score);
}

void ArmSacEngine::chronoFromNow(){
    checkpoint = std::make_shared<std::chrono::time_point<
    std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
}

void ArmSacEngine::logNewGeneration()
{

    std::cout << std::setw(colWidth) << generation << std::setw(colWidth);
    file << std::setw(colWidth) << generation << std::setw(colWidth);

    chronoFromNow();
    
}

void ArmSacEngine::logHeader(){

    // Second line of header
    //*this << std::right;
    std::cout << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Train"<< std::setw(colWidth)<<"Reward";
    file << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Train"<< std::setw(colWidth)<<"Reward";

    if (doValidation) {
        std::cout << std::setw(colWidth) << "Valid" ;
        file << std::setw(colWidth) << "Valid" ;
    }

    if (doTrainingValidation){
        std::cout << std::setw(colWidth) << "TrainVal"<<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
        file << std::setw(colWidth) << "TrainVal"<<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
    }


    std::cout << std::setw(colWidth) << "T_Train"<<std::setw(colWidth)<< "T_Learn";
    file << std::setw(colWidth) << "T_Train"<<std::setw(colWidth)<< "T_Learn";
    if (doValidation) {
        std::cout << std::setw(colWidth) << "T_valid";
        file << std::setw(colWidth) << "T_valid";
    }
    if (doTrainingValidation) {
        std::cout << std::setw(colWidth) << "T_TrainVal";
        file << std::setw(colWidth) << "T_TrainVal";
    }
    std::cout << std::setw(colWidth) << "T_total"<<std::endl;
    file << std::setw(colWidth) << "T_total"<<std::endl;
    
}

void ArmSacEngine::logTraining(double score, double result){

    std::cout<<score<<std::setw(colWidth)<<result<<std::setw(colWidth);
    file<<score<<std::setw(colWidth)<<result<<std::setw(colWidth);

    lastScore = score;
    lastResult = result;

    trainingTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();
}

void ArmSacEngine::logValidation(double score){

    std::cout<<score<<std::setw(colWidth);
    file<<score<<std::setw(colWidth);
    lastValidationScore = score;

    if(lastValidationScore > bestScore){
        bestScore = lastValidationScore;
        learningAgent.saveModels();
    }

    validationTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();




}

void ArmSacEngine::logTrainingValidation(double score){

    std::cout<<score<<std::setw(colWidth);
    file<<score<<std::setw(colWidth);
    lastScore = score; 

    trainingValidationTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
}

void ArmSacEngine::logLimits(){

    std::cout<<armLearnEnv->getCurrentMaxLimitStartingPos()<<std::setw(colWidth)<<armLearnEnv->getCurrentMaxLimitTarget()<<std::setw(colWidth);
    file<<armLearnEnv->getCurrentMaxLimitStartingPos()<<std::setw(colWidth)<<armLearnEnv->getCurrentMaxLimitTarget()<<std::setw(colWidth);
}

void ArmSacEngine::logTimes(){

    totalTime += trainingTime + validationTime + trainingValidationTime;

    std::cout<<trainingTime<<std::setw(colWidth)<<learningTime<<std::setw(colWidth);
    file<<trainingTime<<std::setw(colWidth)<<learningTime<<std::setw(colWidth);
    if(doValidation){
        std::cout<<validationTime<<std::setw(colWidth);
        file<<validationTime<<std::setw(colWidth);
    }
    if(doTrainingValidation){
        std::cout<<trainingValidationTime<<std::setw(colWidth);
        file<<trainingValidationTime<<std::setw(colWidth);
    }
    std::cout<<totalTime<<std::endl;
    file<<totalTime<<std::endl;
}

double ArmSacEngine::getScore(){
    return lastScore;
}

double ArmSacEngine::getResult(){
    return lastResult;
}

torch::Tensor ArmSacEngine::getTensorState(){
    torch::Tensor tensorState = torch::zeros(10, torch::kFloat);

    auto dataSrc = armLearnEnv->getDataSources();
    for(int i=0; i<3;i++){
        tensorState[i] = *dataSrc.at(0).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }

    for(int i=0; i<3;i++){
        tensorState[i+3] = *dataSrc.at(1).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }
        
    for(int i=0; i<4;i++){
        tensorState[i+6] = *dataSrc.at(2).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/4096;
        //std::cout<<*dataSrc.at(2).get().getDataAt(typeid(double), i).getSharedPointer<const double>()<<" ";
    }
    
    return tensorState.view({1, -1});
}