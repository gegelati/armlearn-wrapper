#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <gegelati.h>
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
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", gegelatiParams);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson(ROOT_DIR "/trainParams.json");

    SACParameters sacParams;
    sacParams.loadParametersFromJson(ROOT_DIR "/sacParams.json");

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams);

    // Prompt the number of threads
    std::cout << "Number of threads: " << gegelatiParams.nbThreads << std::endl;

    // Generate validation targets.
    if(gegelatiParams.doValidation){
        armLearnEnv.updateValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    if(trainingParams.progressiveModeTargets){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }


    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, gegelatiParams.maxNbActionsPerEval);
    

#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif

    // init Counter for upgrade the current max limit at 0
    int counterIterationUpgrade = 0;

    // Load the validation trajectories
    armLearnEnv.loadValidationTrajectories()

    // Train for params.nbGenerations generations
    for (int i = 0; i < gegelatiParams.nbGenerations && !exitProgram; i++) {
        armLearnEnv.setgeneration(i);


        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);


        learningAgent.trainOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);

        // Does a validation or not according to the parameter doValidation
        if (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos) {
            std::cout<<" - CurrLimit "<<armLearnEnv.getCurrentMaxLimitTarget()<<std::endl;

            armLearnEnv.updateCurrentLimits(learningAgent.getResult(), gegelatiParams.nbIterationsPerPolicyEvaluation);
        }

        /*if (gegelatiParams.doValidation){

        }*/

        

    }


#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif

    return 0;
}


