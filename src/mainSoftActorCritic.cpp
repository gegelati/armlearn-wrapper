#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
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

void getKey(std::atomic<bool>& exit) {
    std::cout << std::endl;
    std::cout << "Press `q` then [Enter] to exit." << std::endl;
    std::cout.flush();

    exit = false;

    while (!exit) {
        char c;
        std::cin >> c;
        switch (c) {
        case 'q':
        case 'Q':
            exit = true;
            break;
        default:
            printf("Invalid key '%c' pressed.", c);
            std::cout.flush();
        }
    }

    printf("Program will terminate at the end of next generation.\n");
    std::cout.flush();
}

int main() {
    std::cout << "Start ArmLearner application." << std::endl;

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set);

    // Set the parameters from Gegelati.
    // Loads them from "params.json" file
    Learn::LearningParameters gegelatiParams;
    File::ParametersParser::loadParametersFromJson("/params/params.json", gegelatiParams);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson("/params/trainParams.json");

    SACParameters sacParams;
    sacParams.loadParametersFromJson("/params/sacParams.json");

    // Set random seed
    torch::manual_seed(trainingParams.seed);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams);

    // Prompt the number of threads
    std::cout << "Number of threads: " << gegelatiParams.nbThreads << std::endl;

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

    if(trainingParams.interactiveMode){
#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif
    }




    //Creation of the Output stream on cout and on the file
    std::ofstream file("/outLogs/logs.ods", std::ios::out);

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
    for (int i = 0; i < gegelatiParams.nbGenerations && !exitProgram; i++) {
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


    if(trainingParams.interactiveMode){
#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif
    }
    return 0;
}


