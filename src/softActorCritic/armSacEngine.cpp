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
            // Add 1 and multiply by 4.5 return a float between 0 and 9
            // The discrete armLearn has 9 actions
            mulAction = (actionTensor.item<float>() + 1) * 4.5;
            actionTaken = 0;
            
            // Get the action
            while(actionTaken + 1 < mulAction){
                actionTaken++;
            }

            // Do the action
            armLearnEnv->doAction(actionTaken);
        } else {

            auto actionTaken = actionTensor;
            if (!sacParams.continuousActions){
                actionTaken = torch::round(actionTensor * 3 / 2);
            }

            // Convert actionTensor to an actionVector
            std::vector<float> actionVector(actionTaken.data_ptr<float>(), actionTaken.data_ptr<float>() + actionTaken.numel());

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

        //std::cout<<"Action : "<<nbActions<<" - Reward :"<<armLearnEnv->getDistance()<<" - outAction : "<< actionTensor.item<float>()<< "- Acton Send"<< actionTaken<<std::endl;

        // Add the reward to the result
        result += singleReward;
    }

    return result;
}



void ArmSacEngine::trainOneGeneration(uint16_t nbIterationTraining){

    // Log generation
    logNewGeneration();

    double result=0.0;
    double distance=0.0;

    // Train for nbIterationTraining episode(s)
    for(int j = 0; j < sacParams.nbEpisodeTraining; j++){

        uint64_t seed = generation * 100000 + j;
        // Add result
        result += runOneEpisode(seed, Learn::LearningMode::TRAINING, j);
        
        // add distance
        distance += armLearnEnv->getDistance();

        if(armLearnEnv->getDistance() < trainingParams.thresholdUpgrade){
            armLearnEnv->addToDeleteTraj(j);
        }

    }
    // Get the mean distance and result
    result /= sacParams.nbEpisodeTraining;
    distance /= sacParams.nbEpisodeTraining;

    // Log the training
    logTraining(distance, result);

    // Incremente generation
    generation++;

}


void ArmSacEngine::validateOneGeneration(uint16_t nbIterationValidation){

    double result = 0;
    double distance = 0;
    double success = 0;

    // Validate for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationValidation; j++){

        uint64_t seed = generation * 43000000000 + j;
        result += runOneEpisode(seed, Learn::LearningMode::VALIDATION, j);
        // Get distance
        distance += armLearnEnv->getDistance();

        if (armLearnEnv->getDistance() < trainingParams.thresholdUpgrade){
            success++;
        }
    }
    // get the mean result, mean distance and mean success
    result /= nbIterationValidation;
    distance /= nbIterationValidation;
    success /= nbIterationValidation;

    // Log the validation
    logValidation(distance, result, success);
};

void ArmSacEngine::validateTrainingOneGeneration(uint16_t nbIterationTrainingValidation){

    double distance = 0;

    // Validate training for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationTrainingValidation; j++){

        uint64_t seed = generation * 5000000 + j;
        runOneEpisode(seed, Learn::LearningMode::TESTING, j);
        // Get distance
        distance += armLearnEnv->getDistance();
    }
    // get the mean distance
    distance /= nbIterationTrainingValidation;

    // Log the training validation
    logTrainingValidation(distance);
}

void ArmSacEngine::testingModel(uint16_t nbIterationTesting){
    double result = 0;
    double success = 0;

    // Validate for nbIterationTraining episode(s)
    for(int j = 0; j < nbIterationTesting; j++){

        std::cout<<"Episode "<<j+1<<"/"<<nbIterationTesting<<"      "<<std::flush;
        std::cout << '\r';

        uint64_t seed = generation * 43000465132000 + j;
        result += runOneEpisode(seed, Learn::LearningMode::VALIDATION, j);
        // Get distance
        //distance += armLearnEnv->getScore();

        if (armLearnEnv->getDistance() < trainingParams.thresholdUpgrade){
            success++;
        }
    }
    // get the mean result and mean success
    result /= nbIterationTesting;
    success /= nbIterationTesting;

    armLearnEnv->logTestingTrajectories(false);
    

    std::cout<<"Testing resutlt : "<<result<<" -- Testing success rate "<<success<<std::endl;
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

    std::cout << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Tdistance"<< std::setw(colWidth)<<"Treward";
    file << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "Tdistance"<< std::setw(colWidth)<<"Treward";

    if (doValidation) {
        std::cout << std::setw(colWidth) << "Vdistance"<< std::setw(colWidth) << "Vreward" << std::setw(colWidth) << "Success" ;
        file << std::setw(colWidth) << "Vdistance" << std::setw(colWidth) << "Vreward" << std::setw(colWidth) << "Success" ;
    }

    if (doTrainingValidation){
        std::cout << std::setw(colWidth) << "TrainVal";
        file << std::setw(colWidth) << "TrainVal";
    }
    if (doUpdateLimits){
        std::cout<<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
        file <<std::setw(colWidth) << "S_StartP"<<std::setw(colWidth)<<"S_Targ";
    }

    if (trainingParams.controlTrajectoriesDeletion){
        std::cout<<std::setw(colWidth)<<"T_Del";
        file<<std::setw(colWidth)<<"T_Del";
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

void ArmSacEngine::logTraining(double distance, double result){

    std::cout<<distance<<std::setw(colWidth)<<result<<std::setw(colWidth);
    file<<distance<<std::setw(colWidth)<<result<<std::setw(colWidth);

    lastTrainingScore = distance;

    learningAgent.saveModels(generation, false);
    if(distance > bestScore && !doValidation){
        bestScore = distance;
        learningAgent.saveModels(generation, true);
    }

    trainingTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();
}

void ArmSacEngine::logValidation(double distance, double result, double success){

    std::cout<<distance<<std::setw(colWidth)<<result<<std::setw(colWidth)<<success<<std::setw(colWidth);
    file<<distance<<std::setw(colWidth)<<result<<std::setw(colWidth)<<success<<std::setw(colWidth);
    lastValidationScore = success;

    if(lastValidationScore > bestScore){
        bestScore = lastValidationScore;
        learningAgent.saveModels(generation, true);
    }
    validationTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    chronoFromNow();

}

void ArmSacEngine::logTrainingValidation(double distance){

    std::cout<<distance<<std::setw(colWidth);
    file<<distance<<std::setw(colWidth);
    lastTrainingValidationScore = distance; 

    trainingValidationTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
}

void ArmSacEngine::logLimits(){

    std::cout<<armLearnEnv->getCurrentMaxLimitStartingPos()<<std::setw(colWidth)<<armLearnEnv->getCurrentMaxLimitTarget()<<std::setw(colWidth);
    file<<armLearnEnv->getCurrentMaxLimitStartingPos()<<std::setw(colWidth)<<armLearnEnv->getCurrentMaxLimitTarget()<<std::setw(colWidth);
}

void ArmSacEngine::logTrajectoriesDeletion(){
    std::cout<<armLearnEnv->getNbTrajectoriesDeleted()<<std::setw(colWidth);
    file<<armLearnEnv->getNbTrajectoriesDeleted()<<std::setw(colWidth);
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
    torch::Tensor tensorState = torch::zeros(13, torch::kFloat);

    auto dataSrc = armLearnEnv->getDataSources();
    // Get data (cartesian position of the target)
    for(int i=0; i<3;i++){
        tensorState[i] = *dataSrc.at(0).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }

    // Get data (cartesian position of the hand)
    for(int i=0; i<3;i++){
        tensorState[i+3] = *dataSrc.at(1).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }
        
    // Get data (cartesian difference between hand and target)
    for(int i=0; i<3;i++){
        tensorState[i+6] = *dataSrc.at(2).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/100;
    }

    // Get data (angular position of the motors)
    tensorState[9] = *dataSrc.at(3).get().getDataAt(typeid(double), 0).getSharedPointer<const double>()/2048 - 1;
    for(int i=1; i<4;i++){
        tensorState[i+9] = *dataSrc.at(3).get().getDataAt(typeid(double), i).getSharedPointer<const double>()/1024 - 2;
    }
    //std::cout<<tensorState<<std::endl;
    // Flatten the tensor
    return tensorState.view({1, -1});
}