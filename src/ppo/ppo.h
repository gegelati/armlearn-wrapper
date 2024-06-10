#pragma once

#include <torch/torch.h>
#include <random>

#include "networks.h"
#include "ppoParameters.h"
#include "replayBuffer.h"


// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO
{

    private:
        PPOParameters& params;

        ActorCritic net;

        ReplayBuffer buffer;


    public:

        PPO(PPOParameters& params, int stateSize, int actionSize)
        : params(params),
        buffer(),
        net(params.lr, stateSize, actionSize, params.sizeHL1, params.sizeHL2, params.std){
            if(params.loadModels) loadModels();
            
        }

        /**
         * @brief Choose an action with the actor network depending of the observation.
         * The action vector is sample on a Normal distribution
         * 
         * @param observation : Observation tensor
         * 
         * @return Action taken
         */
        std::pair<at::Tensor, at::Tensor> chooseAction(torch::Tensor observation);

        /**
         * @brief Save data in the memory buffer
         * 
         * @param state Last State
         * @param action Last action
         * @param reward Last reward
         * @param newState Last new State
         * @param done Last done
         */
        void remember(torch::Tensor state, std::pair<torch::Tensor, torch::Tensor> actionNCritic, double reward, bool done);

        std::vector<torch::Tensor> expectedReturns(std::vector<torch::Tensor> rewards, std::vector<torch::Tensor> dones, std::vector<torch::Tensor> vals);

        /// @brief Learn function
        void learn();

        /// @brief Load saved models, throw exception if models have not been saved before
        void loadModels();

        /** 
         * @brief save models
         * 
         * @param generationNumber the integer corresponding to the generation
         * @param bestModel boolean that indicate if this is the best model
         */
        void saveModels(uint64_t genenrationNumber, bool bestModel=false);

        int getBufferSize();

};