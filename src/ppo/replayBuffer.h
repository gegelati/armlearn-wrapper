
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <torch/torch.h>
#include <vector>


class ReplayBuffer{
    private:

        /// Memory of states 
        std::vector<torch::Tensor> stateMemory;

        /// Memory of actions 
        std::vector<torch::Tensor> actionMemory;

        /// Memory of rewards 
        std::vector<torch::Tensor> rewardMemory;

        /// Memory of terminal
        std::vector<torch::Tensor> terminalMemory;

        /// Memory of value
        std::vector<torch::Tensor> valueMemory;

        /// Memory of logProbs
        std::vector<torch::Tensor> logProbsMemory;


    public:

        /**
         * @brief Save data in the different tensor
         * 
         * @param state Last State
         * @param action Last action
         * @param reward Last reward
         * @param done Last done
         * @param value Last value
         * @param logProbs last log prob
         */
        void storeTransition(torch::Tensor state, torch::Tensor action, double reward, bool done, torch::Tensor value, torch::Tensor logProbs);

        void storeValue(torch::Tensor value);

        void clearData();

        /// Return states
        std::vector<torch::Tensor> getStateMemory();
        
        /// Return actions
        std::vector<torch::Tensor> getActionMemory();

        /// Return rewards
        std::vector<torch::Tensor> getRewardMemory();

        /// Return terminals
        std::vector<torch::Tensor> getTerminalMemory();

        /// Return value
        std::vector<torch::Tensor> getValueMemory();

        /// Return log probs
        std::vector<torch::Tensor> getLogProbsMemory();

};

#endif