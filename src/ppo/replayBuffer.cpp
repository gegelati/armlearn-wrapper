#include <algorithm>
#include <torch/torch.h>

#include "replayBuffer.h"


void ReplayBuffer::storeTransition(torch::Tensor state, torch::Tensor action, double reward, 
                                   bool done, torch::Tensor value, torch::Tensor logProbs){
    
    // Store the data
    stateMemory.push_back(state);
    actionMemory.push_back(action);
    rewardMemory.push_back(torch::tensor({reward}, torch::kFloat));
    terminalMemory.push_back(torch::tensor({done}, torch::kInt));
    valueMemory.push_back(value);
    logProbsMemory.push_back(logProbs);
}

void ReplayBuffer::storeValue(torch::Tensor value){
    valueMemory.push_back(value);
}

void ReplayBuffer::clearData(){
    stateMemory.clear();
    actionMemory.clear();
    rewardMemory.clear();
    terminalMemory.clear();
    valueMemory.clear();
    logProbsMemory.clear();
}

std::vector<torch::Tensor> ReplayBuffer::getStateMemory(){
    return stateMemory;
}

/// Return actions
std::vector<torch::Tensor> ReplayBuffer::getActionMemory(){
    return actionMemory;
}

/// Return rewards
std::vector<torch::Tensor> ReplayBuffer::getRewardMemory(){
    return rewardMemory;
}

/// Return terminals
std::vector<torch::Tensor> ReplayBuffer::getTerminalMemory(){
    return terminalMemory;
}

/// Return value
std::vector<torch::Tensor> ReplayBuffer::getValueMemory(){
    return valueMemory;
}

/// Return log probs
std::vector<torch::Tensor> ReplayBuffer::getLogProbsMemory(){
    return logProbsMemory;
}