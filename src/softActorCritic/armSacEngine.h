

#ifndef ARM_SAC_ENGINE_H
#define ARM_SAC_ENGINE_H


#include "softActorCritic.h"
#include "sacParameters.h"
#include "../ArmLearnWrapper.h"
#include <torch/torch.h>

class ArmSacEngine{
    private:
        /// Soft Actor Critic Agent
        SoftActorCritic learningAgent;

        /// Parameters for the soft actor critic
        SACParameters sacParams;

        /// ArmLearn Environnement
        ArmLearnWrapper* armLearnEnv;

        /// Current Generation
        uint16_t generation=0;

        /// Max number of actions doable in one episode
        uint16_t maxNbActions;

        /// True if validation is done
        bool doValidation;

        /// True if training validation is done
        bool doTrainingValidation;

        /// True if limits are updated
        bool doUpdateLimits;

        /// Best score gotten in a validation cycle, if doValidation is false, it is the one gotten in a training cycle
        double bestScore = -10000;

        double lastTrainingScore;

        /// Last score gotten in validation
        double lastValidationScore;

        /// Last score gotten in trainingValidation
        double lastTrainingValidationScore;

        /// Time of learning
        double learningTime=0;
        /// Time of training
        double trainingTime=0;
        /// Time of validation
        double validationTime=0;
        /// Time of training validation
        double trainingValidationTime=0;
        /// Total time
        double totalTime = 0;

        /// Log file
        std::ostream& file;

        /// column to space for the logs
        uint16_t colWidth = 12;

        /// Checkpoint of time
        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;

        //std::vector<std::vector<int16_t>> vectorValue;

    public:
        ArmSacEngine(SACParameters sacParams, ArmLearnWrapper* armLearnEnv, std::ostream& file,
        uint16_t maxNbActions, bool doValidation=false, bool doTrainingValidation=false, bool doUpdateLimits=false)
        : sacParams(sacParams), file(file), armLearnEnv(armLearnEnv), learningAgent(sacParams, 10, (sacParams.multipleActions) ? 4:1) {

            this->maxNbActions = maxNbActions;
            this->doValidation=doValidation;
            this->doTrainingValidation=doTrainingValidation;
            this->doUpdateLimits=doUpdateLimits;
            this->logHeader();
            file.flush();
            
            
        }

        /**
         * @brief run one episode
         * 
         * @param[in] seed Seed of the instance
         * @param[in] mode current mode (training, validation or testing)
         * @param[in] iterationNumber current iteration Number
         */
        double runOneEpisode(uint16_t seed, Learn::LearningMode mode, uint16_t iterationNumber);

        /**
         * @brief Train one generation
         * 
         * @param[in] nbIterationTraining Number of episode to train on
         */
        void trainOneGeneration(uint16_t nbIterationTraining);

        /**
         * @brief Validate one generation
         * 
         * @param[in] nbIterationTraining Number of episode to validate on
         */
        void validateOneGeneration(uint16_t nbIterationValidation);

        /**
         * @brief validate training one generation
         * 
         * @param[in] nbIterationTraining Number of episode to validate training on
         */
        void validateTrainingOneGeneration(uint16_t nbIterationTrainingValidation);

        /// Save current time
        void chronoFromNow();

        /// Print the headers of the logs
        void logHeader();

        /// Log the generation and reset checkpoint time
        void logNewGeneration();

        /// Log the training result and score, potentially save the models
        void logTraining(double score, double result);

        /// Log the validation score, potentially save the models
        void logValidation(double score, double success);

        /// Log the training validation score
        void logTrainingValidation(double score);
        
        /// Log the size limits of the environnement (targets and starting position)
        void logLimits();

        /// Log the training, learning, validation, training validation and total times
        void logTimes();

        /// Convert the state from the environnement to a tensor
        torch::Tensor getTensorState();

        /// Return lastTrainingScore 
        double getLastTrainingScore();

        /// Return lastTrainingValidationScore
        double getLastTrainingValidationScore();

};

#endif