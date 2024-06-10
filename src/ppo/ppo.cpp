#include <torch/torch.h>
#include <filesystem>
#include <iostream>

#include "ppo.h"

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());

std::pair<at::Tensor, at::Tensor> PPO::chooseAction(torch::Tensor observation){

    // Get the action and return it
    return net.forward(observation);
}

void PPO::remember(torch::Tensor state, std::pair<torch::Tensor, torch::Tensor> actionNCritic, double reward, bool done){
    buffer.storeTransition(
        state, 
        actionNCritic.first, 
        reward, 
        done,
        actionNCritic.second,
        net.log_prob(actionNCritic.first)
        );
}

std::vector<torch::Tensor> PPO::expectedReturns(std::vector<torch::Tensor> rewards, std::vector<torch::Tensor> dones, std::vector<torch::Tensor> vals){
    // Compute the returns.
    torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
    std::vector<torch::Tensor> returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

    for (uint64_t i=rewards.size(); i-->0;) // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
    {
        // Advantage.
        auto delta = rewards[i] + params.gamma*vals[i+1]*(1-dones[i]) - vals[i];
        gae = delta + params.gamma*params.lambda*(1-dones[i])*gae;
        returns[i] = gae + vals[i];
    }
    return returns;
}

void PPO::learn(){
    
    buffer.storeValue(net.forward(buffer.getStateMemory()[params.sizeBuffer-1]).second);

    torch::Tensor log_probs = torch::cat(buffer.getLogProbsMemory()).detach();
    torch::Tensor returns = torch::cat(expectedReturns(buffer.getRewardMemory(), buffer.getTerminalMemory(), buffer.getValueMemory())).detach();
    torch::Tensor values = torch::cat(buffer.getLogProbsMemory()).detach();
    torch::Tensor actions = torch::cat(buffer.getActionMemory());
    torch::Tensor states = torch::cat(buffer.getStateMemory());
    torch::Tensor rewards = torch::cat(buffer.getRewardMemory());


    torch::Tensor advantages = returns - values.slice(0, 0, params.sizeBuffer);



    for (int e = 0; e<params.nbEpochs; e++)
    {
        // Generate random indices.
        torch::Tensor cpy_sta = torch::zeros({params.batchSize, states.size(1)}, states.options());
        torch::Tensor cpy_act = torch::zeros({params.batchSize, actions.size(1)}, actions.options());
        torch::Tensor cpy_log = torch::zeros({params.batchSize, log_probs.size(1)}, log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({params.batchSize, returns.size(1)}, returns.options());
        torch::Tensor cpy_adv = torch::zeros({params.batchSize, advantages.size(1)}, advantages.options());

        for (int b=0;b<params.batchSize;b++) {

            int idx = std::uniform_int_distribution<int>(0, params.sizeBuffer-1)(re);
            cpy_sta[b] = states[idx];
            cpy_act[b] = actions[idx];
            cpy_log[b] = log_probs[idx];
            cpy_ret[b] = returns[idx];
            cpy_adv[b] = advantages[idx];
        }
        

        auto av = net.forward(cpy_sta); // action value pairs
        auto action = av.first;
        auto entropy = net.entropy().mean();
        auto new_log_prob = net.log_prob(cpy_act);

        auto old_log_prob = cpy_log;
        auto ratio = (new_log_prob - old_log_prob).exp();
        auto surr1 = ratio*cpy_adv;
        auto surr2 = torch::clamp(ratio, 1. - params.clip, 1. + params.clip)*cpy_adv;

        auto val = av.second;
        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto critic_loss = (cpy_ret-val).pow(2).mean();

        auto loss = 0.5*critic_loss+actor_loss-params.beta*entropy;

        net.getPtrOptimizer()->zero_grad();
        loss.backward();
        net.getPtrOptimizer()->step();
    }

    buffer.clearData();

}

void PPO::loadModels(){
    std::cout<<" ----- Loading Models ----- "<<std::endl;
    std::string slashToAdd = (std::filesystem::exists(params.pathModel)) ? "": "/";
    std::string path = (slashToAdd + params.pathModel + "/models/best_model/").c_str();
    net.loadCheckpoint(path);
}

void PPO::saveModels(uint64_t genenrationNumber, bool bestModel){
    //std::cout<<" ----- Saving Models ----- "<<std::endl;
    std::string slashToAdd = (std::filesystem::exists(params.pathModel)) ? "": "/";
    std::string path = (slashToAdd + params.pathModel + "/models/save_" + std::to_string(genenrationNumber) + "/").c_str();
    if(bestModel){
        path = (slashToAdd + params.pathModel + "/models/best_model/").c_str();
    }
    if(!std::filesystem::exists(path)){
        std::filesystem::create_directory(path);
    }
    net.saveCheckpoint(path);
}

int PPO::getBufferSize(){
    return buffer.getStateMemory().size();
}

    
