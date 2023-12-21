#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <gegelati.h>
#include <torch/torch.h>
#include "instructions.h"
#include "trainingParameters.h"
#include "armLearnLogger.h"

#include "ArmLearnWrapper.h"
#include "softActorCritic/armSacEngine.h"
#include "softActorCritic/sacParameters.h"

int main() {
    std::cout << "Start ArmLearner application." << std::endl;

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set);

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

    // Set random seed
    torch::manual_seed(trainingParams.seed);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams);

    // Prompt the number of threads and set it to torch
    std::cout << "Number of threads: " << gegelatiParams.nbThreads << std::endl;
    torch::set_num_threads(gegelatiParams.nbThreads);

    // If a validation target is done
    bool doTrainingValidation = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);

    // Generate validation targets.
    if(gegelatiParams.doValidation){
        armLearnEnv.updateValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    if(doTrainingValidation){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    std::atomic<bool> exitProgram = false; // (set to false by other thread)
    std::thread threadKeyboard;




    //Creation of the Output stream on cout and on the file
    std::ofstream file((slashToAdd + "outLogs/logs.ods").c_str(), std::ios::out);

    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, file, gegelatiParams.maxNbActionsPerEval, 
                               gegelatiParams.doValidation, doTrainingValidation);

    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories();
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories();
    }

    // Train for params.nbGenerations generations
    for (int i = 0; i < gegelatiParams.nbGenerations; i++) {
        armLearnEnv.setgeneration(i);

        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);


        // Train
        learningAgent.trainOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);

        // Does a validation or not according to the parameter doValidation
        if (gegelatiParams.doValidation){
            learningAgent.validateOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);
        }

        // Does a training validation or not according to doTrainingValidation
        if (doTrainingValidation) {
            learningAgent.validateTrainingOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);

            // Log the limits
            learningAgent.logLimits();

            // Update limits
            armLearnEnv.updateCurrentLimits(learningAgent.getLastTrainingValidationScore(), gegelatiParams.nbIterationsPerPolicyEvaluation);

        }
        learningAgent.logTimes();

    }

    return 0;
}


