#include "networks.h"


torch::optim::Adam* ActorCritic::getPtrOptimizer(){
    return &optimizer;
}

// Forward pass.
std::pair<at::Tensor, at::Tensor> ActorCritic::forward(torch::Tensor x)
{

    // Actor.
    mu_ = torch::relu(a_lin1_->forward(x));
    mu_ = torch::relu(a_lin2_->forward(mu_));
    mu_ = torch::tanh(a_lin3_->forward(mu_));

    // Critic.
    torch::Tensor val = torch::relu(c_lin1_->forward(x));
    val = torch::relu(c_lin2_->forward(val));
    val = torch::tanh(c_lin3_->forward(val));
    val = c_val_->forward(val);

    if (this->is_training()) 
    {
        torch::NoGradGuard no_grad;

        torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
        return std::make_pair(action, val);  
    }
    else 
    {
        return std::make_pair(mu_, val);  
    }
}

torch::Tensor ActorCritic::entropy()
{
    // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
    return 0.5 + 0.5*log(2*M_PI) + log_std_;
}

torch::Tensor ActorCritic::log_prob(torch::Tensor action)
{
    // Logarithmic probability of taken action, given the current distribution.
    torch::Tensor var = (log_std_+log_std_).exp();

    return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
}

void ActorCritic::loadCheckpoint(std::string path){
    torch::serialize::InputArchive inputArchive;
    inputArchive.load_from(path + "model.pt");
    load(inputArchive);
}

void ActorCritic::saveCheckpoint(std::string path){
    torch::serialize::OutputArchive outputArchive;
    save(outputArchive);
    outputArchive.save_to(path + "model.pt");
}