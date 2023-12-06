

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

        bool doValidation;
        bool doTrainingValidation;

        double bestResult = -10000;
        double bestScore = -10000;
        double lastValidationScore;
        double lastResult;
        double lastScore;

        double learningTime=0;
        double trainingTime=0;
        double validationTime=0;
        double trainingValidationTime=0;
        double totalTime = 0;

        std::ostream& file;

        uint16_t colWidth = 12;
        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;

    public:
        ArmSacEngine(SACParameters sacParams, ArmLearnWrapper* armLearnEnv, std::ostream& file,
        uint16_t maxNbActions, bool doValidation=false, bool doTrainingValidation=false)
        : sacParams(sacParams), file(file),
        learningAgent(sacParams, 10, 4),
        armLearnEnv(armLearnEnv) {
            this->maxNbActions = maxNbActions;
            this->doValidation=doValidation;
            this->doTrainingValidation=doTrainingValidation;
            this->logHeader();
            file.flush();
            
            
        }

        double runOneEpisode(uint16_t seed, Learn::LearningMode mode);

        void trainOneGeneration(uint16_t nbIterationTraining);

        void validateOneGeneration(uint16_t nbIterationValidation);

        void validateTrainingOneGeneration(uint16_t nbIterationTrainingValidation);


        void chronoFromNow();

        void logHeader();

        void logNewGeneration();

        void logTraining(double score, double result);

        void logValidation(double score);

        void logTrainingValidation(double score);
        
        void logLimits();

        void logTimes();

        torch::Tensor getTensorState();

        double getScore();

        double getResult();




};

#endif