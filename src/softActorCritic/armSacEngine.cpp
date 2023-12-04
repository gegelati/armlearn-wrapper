#include <torch/torch.h>

#include "armSacEngine.h"


double ArmSacEngine::runOneEpisode(uint16_t seed, bool training){

    torch::Tensor newState;
    torch::Tensor actionTensor;
    float mulAction;
    uint64_t actionTaken;
    double singleReward;
    double result=0;

    armLearnEnv->reset(seed, Learn::LearningMode::TRAINING);
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

        /*std::vector<float> stateVector(state.data_ptr<float>(), state.data_ptr<float>() + state.numel());

        std::cout<<"Actions ";
        for(auto val : actionVector){
            std::cout<<val<<" ";
        }

        std::cout<<"- State ";
        for(auto val : stateVector){
            std::cout<<val<<" ";
        }*/

        armLearnEnv->doActionContinuous(actionVector);
        newState = getTensorState();
        
        singleReward = armLearnEnv->getReward();
        
        //std::cout<<"- Reward "<<singleReward<<std::endl;
        nbActions++;
        terminated = (armLearnEnv->isTerminal() || nbActions == maxNbActions);

        if(training){
            if(nbActions < maxNbActions)
                learningAgent.remember(state, actionTensor, singleReward, newState, terminated);

            learningAgent.learn();
        }


        state = newState;
        result += singleReward;
    }
    return result;
}

void ArmSacEngine::trainOneGeneration(uint16_t nbIterationTraining){

    double result = 0.0;
    double score = 0.0;
    double singleReward = 0.0;
    
    // we train the TPG, see doaction for the reward function
    for(int j = 0; j < nbIterationTraining; j++){

        uint64_t seed = generation * 100000 + j;
        result += runOneEpisode(seed, true);
        score += armLearnEnv->getScore();
    }
    result = result / nbIterationTraining;
    score = score / nbIterationTraining;
    if(score > -5){
        learningAgent.saveModels();
    }

    memoryResult.push_back(result);
    memoryScore.push_back(score);
    auto moyenneReward = std::accumulate(memoryResult.begin(), memoryResult.end(), 0.0) / memoryResult.size();
    auto moyenneScore = std::accumulate(memoryScore.begin(), memoryScore.end(), 0.0) / memoryScore.size();
    
    auto recentMoyenneReward = (memoryResult.size()>10) ? std::accumulate(memoryResult.end()-10, memoryResult.end(), 0.0) / 10 : moyenneReward;

    std::cout<<"Gen "<<generation<<" - Result "<<result<<" - Score "<<score;
    std::cout<<" - Mean Result "<<moyenneReward<<" - Mean Result Last 10 "<<recentMoyenneReward<<" - Mean Score "<<moyenneScore;

    generation++;
    lastResult = result;
    lastScore = score;
    
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
    //std::cout<<std::endl;
    
    return tensorState.view({1, -1});
}