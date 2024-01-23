#include <torch/torch.h>
#include <algorithm>
#include <numeric>

#include "armSacEngine.h"


double ArmSacEngine::runOneEpisode(uint16_t seed, Learn::LearningMode mode, uint16_t iterationNumber){

    // Create variables
    torch::Tensor newState;
    torch::Tensor actionTensor;
    float mulAction;
    uint64_t actionTaken;
    double singleReward;
    double result=0;
    uint64_t nbActions = 0;
        
    // Reset the environnement
    armLearnEnv->reset(seed, mode, iterationNumber);

    // Set terminated to false and get the state
    bool terminated = false;
    torch::Tensor state = getTensorState();


    // Do iterations while the episode is not terminated
    while (!terminated && nbActions < maxNbActions) {

        // Get the continuous action
        actionTensor = learningAgent.chooseAction(state);

        if(!sacParams.multipleActions){
            // actionTensor return float between -1 and 1. 
            // Add 1 and multiply by 4 return a float between 0 and 8
            // The discrete armLearn has 9 actions
            mulAction = (actionTensor.item<float>() + 1) * 4;
            actionTaken = 0;
            
            // Get the action
            while(actionTaken + 1 < mulAction){
                actionTaken++;
            }

            // Do the action
            armLearnEnv->doAction(actionTaken);
        } else {

            if (!sacParams.continuousActions){
                actionTensor = torch::round(actionTensor);
            }

            // Convert actionTensor to an actionVector
            std::vector<float> actionVector(actionTensor.data_ptr<float>(), actionTensor.data_ptr<float>() + actionTensor.numel());

            // Do a continuous action
            armLearnEnv->doActionContinuous(actionVector);

        }



        // Get the new state
        newState = getTensorState();
        
        // Get the reward
        singleReward = armLearnEnv->getReward();
        
        // Incremente action counter
        nbActions++;

        // Get terminated state
        terminated = armLearnEnv->isTerminal();

        // if mode is training mode, train..
        if(mode == Learn::LearningMode::TRAINING){
            // Get time
            auto time = std::make_shared<std::chrono::time_point<
            std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());

            // Remember data
            learningAgent.remember(state, actionTensor, singleReward, newState, terminated);

            // Learn
            learningAgent.learn();

            // Add this learning time to the learning time
            learningTime += ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *time)).count();
        }

        // Set state to newState
        state = newState;

        // Add the reward to the result
        result += singleReward;
    }

    return result;
}



void ArmSacEngine::trainOneGeneration(uint16_t nbIterationTraining){

    // Log generation
    logNewGeneration();

    double result=0.0;
    double score=0.0;

    // Train for nbIterationTraining episode(s)
    for(int j = 0; j < sacParams.nbEpisodeTraining; j++){

        uint64_t seed = generation * 100000 + j;
        // Add result
        result += runOneEpisode(seed, Learn::LearningMode::TRAINING, j);
        
        // add score
        score += armLearnEnv->getScore();

        if(armLearnEnv->getScore() > -5){
            armLearnEnv->addToDeleteTraj(j);
        }

    }
    // Get the mean score and result
    result /= sacParams.nbEpisodeTraining;
    score /= sacParams.nbEpisodeTraining;

    // Log the training
    logTraining(score, result);

    // Incremente generation
    generation++;

}


void ArmSacEngine::validateOneGeneration(uint16_t nbIterationValidation){

    double score = 0;
    double success = 0;

    // Validate for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationValidation; j++){

        uint64_t seed = generation * 43000000000 + j;
        score += runOneEpisode(seed, Learn::LearningMode::VALIDATION, j);
        // Get score
        //score += armLearnEnv->getScore();

        if (armLearnEnv->getReward() > 0){
            success++;
        }
    }
    // get the mean score and mean success
    score /= nbIterationValidation;
    success /= nbIterationValidation;

    // Log the validation
    logValidation(score, success);
};

void ArmSacEngine::validateTrainingOneGeneration(uint16_t nbIterationTrainingValidation){

    double score = 0;

    // Validate training for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationTrainingValidation; j++){

        uint64_t seed = generation * 5000000 + j;
        runOneEpisode(seed, Learn::LearningMode::TESTING, j);
        // Get score
        score += armLearnEnv->getScore();
    }
    // get the mean score
    score /= nbIterationTrainingValidation;

    // Log the training validation
    logTrainingValidation(score);
}

void ArmSacEngine::testingModel(uint16_t nbIterationTesting){
    double score = 0;
    double success = 0;

    // Validate for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationTesting; j++){

        uint64_t seed = generation * 43000465132000 + j;
        score += runOneEpisode(seed, Learn::LearningMode::VALIDATION, j);
        // Get score
        //score += armLearnEnv->getScore();

        if (armLearnEnv->getReward() > 0){
            success++;
        }
    }
    // get the mean score and mean success
    score /= nbIterationTesting;
    success /= nbIterationTesting;

    armLearnEnv->logTestingTrajectories();
    

    std::cout<<"Testing score : "<<score<<" -- Testing success rate "<<success<<std::endl;
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

    std::cout << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Train"<< std::setw(colWidth)<<"Reward";
    file << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Train"<< std::setw(colWidth)<<"Reward";

    if (doValidation) {
        std::cout << std::setw(colWidth) << "Valid"<< std::setw(colWidth) << "Success" ;
        file << std::setw(colWidth) << "Valid" << std::setw(colWidth) << "Success" ;
    }

    if (doTrainingValidation){
        std::cout << std::setw(colWidth) << "TrainVal";
        file << std::setw(colWidth) << "TrainVal";
    }
    if (doUpdateLimits){
        std::cout<<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
        file <<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
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

    lastTrainingScore = score;

    learningAgent.saveModels(generation, false);
    if(score > bestScore && !doValidation){
        bestScore = score;
        learningAgent.saveModels(generation, true);
    }

    trainingTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();
}

void ArmSacEngine::logValidation(double score, double success){

    std::cout<<score<<std::setw(colWidth)<<success<<std::setw(colWidth);
    file<<score<<std::setw(colWidth)<<success<<std::setw(colWidth);
    lastValidationScore = score;

    if(lastValidationScore > bestScore){
        bestScore = lastValidationScore;
        learningAgent.saveModels(generation, true);
    }
    validationTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();

}

void ArmSacEngine::logTrainingValidation(double score){

    std::cout<<score<<std::setw(colWidth);
    file<<score<<std::setw(colWidth);
    lastTrainingValidationScore = score; 

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

    trainingTime = 0;
    learningTime = 0;
    validationTime = 0;
    trainingValidationTime = 0;
}

double ArmSacEngine::getLastTrainingValidationScore(){
    return lastTrainingValidationScore;
}

double ArmSacEngine::getLastTrainingScore(){
    return lastTrainingScore;
}

torch::Tensor ArmSacEngine::getTensorState(){
    // Create zero tensor
    torch::Tensor tensorState = torch::zeros(10, torch::kFloat);

    auto dataSrc = armLearnEnv->getDataSources();
    // Get data (cartesian position of the hand)
    for(int i=0; i<3;i++){
        tensorState[i] = *dataSrc.at(0).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }

    // Get data (cartesian difference between hand and target)
    for(int i=0; i<3;i++){
        tensorState[i+3] = *dataSrc.at(1).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }
        
    // Get data (angular position of the motors)
    for(int i=0; i<4;i++){
        tensorState[i+6] = *dataSrc.at(2).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/4096;
    }
    
    // Flatten the tensor
    return tensorState.view({1, -1});
}