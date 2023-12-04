

#ifndef ARM_SAC_ENGINE_H
#define ARM_SAC_ENGINE_H

#include <torch/torch.h>

#include "softActorCritic.h"
#include "sacParameters.h"
#include "../ArmLearnWrapper.h"

class ArmSacEngine{
    private:
        SoftActorCritic learningAgent;
        SACParameters sacParams;
        ArmLearnWrapper* armLearnEnv;

        uint16_t generation=0;
        uint16_t maxNbActions;

        std::vector<double> memoryResult;
        std::vector<double> memoryScore;

        double bestResult = -10000;
        double lastResult;
        double lastScore;

    public:
        ArmSacEngine(SACParameters sacParams, ArmLearnWrapper* armLearnEnv, uint16_t maxNbActions)
        : sacParams(sacParams),
        learningAgent(sacParams, 10, 4),
        armLearnEnv(armLearnEnv) {
            this->maxNbActions = maxNbActions;
        }

        void trainOneGeneration(uint16_t nbIterationTraining);

        double runOneEpisode(uint16_t seed, bool training=true);

        torch::Tensor getTensorState();

        double getScore();

        double getResult();




};

#endif