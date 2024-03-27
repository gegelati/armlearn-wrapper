#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <gegelati.h>
#include "trainingParameters.h"
#include "armLearnLogger.h"

#include "ArmLearnWrapper.h"
#include "softActorCritic/armSacEngine.h"
#include "softActorCritic/sacParameters.h"
#include <torch/torch.h>

int main() {
    std::cout << "Start ArmLearner application." << std::endl;

    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";

    // Set the parameters from Gegelati.
    // Loads them from "params.json" file
    Learn::LearningParameters gegelatiParams;
    File::ParametersParser::loadParametersFromJson((slashToAdd + "params/params.json").c_str(), gegelatiParams);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((slashToAdd + "params/trainParams.json").c_str());

    SACParameters sacParams;
    sacParams.loadParametersFromJson((slashToAdd + "params/sacParams.json").c_str());


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams, false);

    // Set and Prompt the number of threads
    torch::set_num_threads(gegelatiParams.nbThreads);

    // Set random seed
    torch::manual_seed(trainingParams.seed);
    std::cout << "Number of threads: " << torch::get_num_threads() << std::endl;


    // If a validation target is done
    bool doUpdateLimits = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);
    bool doTrainingValidation = (trainingParams.doTrainingValidation && doUpdateLimits);

    // Generate validation targets.
    if(gegelatiParams.doValidation && !trainingParams.loadValidationTrajectories){
        armLearnEnv.updateValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    if(doTrainingValidation){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }



    //Creation of the Output stream on cout and on the file
    auto nameLogs = (!trainingParams.testing) ? "logsSAC" : "garbage";
    std::ofstream file((slashToAdd + "outLogs/" + nameLogs + ".ods").c_str(), std::ios::out);

    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, file, trainingParams, gegelatiParams.maxNbActionsPerEval, 
                               gegelatiParams.doValidation);

    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories();
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories();
    }

    if(trainingParams.testing){
        learningAgent.testingModel(gegelatiParams.nbIterationsPerPolicyEvaluation);
    } else {

        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
        std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
        bool timeLimitReached = false;

        // Train for params.nbGenerations generations
        for (int i = 0; i < gegelatiParams.nbGenerations && !timeLimitReached; i++) {
            armLearnEnv.setgeneration(i);

            // Update/Generate the training trajectories
            armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);

            // Train
            learningAgent.trainOneGeneration(trainingParams.nbIterationTraining);

            // Does a validation or not according to the parameter doValidation
            if (gegelatiParams.doValidation){
                learningAgent.validateOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);
            }

            // Does a training validation or not according to doTrainingValidation
            if (doTrainingValidation) {
                learningAgent.validateTrainingOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);
            }

            if (doUpdateLimits) {

                // Log the limits
                learningAgent.logLimits();

                // Update limits
                if (doTrainingValidation) {
                    armLearnEnv.updateCurrentLimits(learningAgent.getLastTrainingValidationScore(), gegelatiParams.nbIterationsPerPolicyEvaluation);
                }
                else{
                    armLearnEnv.updateCurrentLimits(learningAgent.getLastTrainingScore(), gegelatiParams.nbIterationsPerPolicyEvaluation);
                }
                
            }

            learningAgent.logTimes();

            // Check time limit only if the parameter is above 0
            if(trainingParams.timeMaxTraining > 0){
                // Set true if the time is above the limit
                timeLimitReached = (((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count() > trainingParams.timeMaxTraining);
            }
    }


    }

    return 0;
}


