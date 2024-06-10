#pragma once

#include <torch/torch.h>
#include <math.h>

// Network model for Proximal Policy Optimization on Incy Wincy.
class ActorCritic : public torch::nn::Module 
{
    private:
        // Actor.
        torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
        torch::Tensor mu_;
        torch::Tensor log_std_;

        // Critic.
        torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

        
        torch::optim::Adam optimizer;

    public:
        ActorCritic(double lr, int stateSize, int actionSize, int sizeHL1=256, int sizeHL2=256, double std=2e-2)
            : a_lin1_(register_module("a_lin1", torch::nn::Linear(stateSize, sizeHL1))),
            a_lin2_(register_module("a_lin2", torch::nn::Linear(sizeHL1, sizeHL2))),
            a_lin3_(register_module("a_lin3", torch::nn::Linear(sizeHL2, actionSize))),
            mu_(torch::full(actionSize, 0.)),
            log_std_(torch::full(actionSize, std)),
            
            c_lin1_(register_module("c_lin1", torch::nn::Linear(stateSize, sizeHL1))),
            c_lin2_(register_module("c_lin2", torch::nn::Linear(sizeHL1, sizeHL2))),
            c_lin3_(register_module("c_lin3", torch::nn::Linear(sizeHL2, actionSize))),
            c_val_(register_module("c_val", torch::nn::Linear(actionSize, 1))),
            optimizer(this->parameters(), torch::optim::AdamOptions(lr))
        {
        }

        // Forward pass.
        std::pair<at::Tensor, at::Tensor> forward(torch::Tensor x);

        torch::Tensor entropy();

        torch::Tensor log_prob(torch::Tensor action);

        /// Return a pointer to the optimizer
        torch::optim::Adam* getPtrOptimizer();

        /// Load model
        void loadCheckpoint(std::string path);

        /// Save model
        void saveCheckpoint(std::string path);
};
