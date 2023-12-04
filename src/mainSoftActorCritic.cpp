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
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams.coefRewardNbIterations);

    // Prompt the number of threads
    std::cout << "Number of threads: " << gegelatiParams.nbThreads << std::endl;


    // If the training is progressive, set the limit to the param value
    if (trainingParams.progressiveModeTargets) armLearnEnv.setCurrentMaxLimitTarget(trainingParams.maxLengthTargets);


    // If the training is progressive, set the limit to the param value
    if (trainingParams.progressiveModeStartingPos) armLearnEnv.setCurrentMaxLimitStartingPos(trainingParams.maxLengthStartingPos);

    // Generate validation targets.
    if(gegelatiParams.doValidation){
        armLearnEnv.updateValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    if(trainingParams.progressiveModeTargets){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(
            gegelatiParams.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);
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

    // Train for params.nbGenerations generations
    for (int i = 0; i < gegelatiParams.nbGenerations && !exitProgram; i++) {
        armLearnEnv.setgeneration(i);


        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(
            gegelatiParams.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);


        learningAgent.trainOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);

        // Does a validation or not according to the parameter doValidation
        if (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos) {
            std::cout<<" - CurrLimit "<<armLearnEnv.getCurrentMaxLimitTarget()<<std::endl;
            /*auto validationResults =
                la.evaluateAllRoots(i, Learn::LearningMode::TESTING);
            
            // Get the best result of the training validation
            auto iter = validationResults.begin();
            std::advance(iter, validationResults.size() - 1);
            double bestResult = iter->first->getResult();*/

            // If the best TPG is above the threshold for upgrade
            if (learningAgent.getResult() > 0){

                // Incremente the counter for upgrading the max current limit
                counterIterationUpgrade += 1;

                // If the counter reach the number of iterations to upgrade
                if(counterIterationUpgrade == trainingParams.nbIterationsUpgrade){

                    // Upgrade the limit of tagets
                    if (trainingParams.progressiveModeTargets){
                        auto currentMaxLimitTarget = std::min(armLearnEnv.getCurrentMaxLimitTarget() * trainingParams.coefficientUpgrade, 1000.0d);
                        armLearnEnv.setCurrentMaxLimitTarget(currentMaxLimitTarget);
                    }


                    // Upgrade the limit of starting positions
                    if (trainingParams.progressiveModeStartingPos){
                        auto currentMaxLimitStartingPos = std::min(armLearnEnv.getCurrentMaxLimitStartingPos() * trainingParams.coefficientUpgrade, 200.0d);
                        armLearnEnv.setCurrentMaxLimitStartingPos(currentMaxLimitStartingPos);
                    }


                    counterIterationUpgrade = 0;

                    // Update the training validation trajectories
                    armLearnEnv.updateTrainingValidationTrajectories(
                        gegelatiParams.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);
                }
            }
            // Reset the counter
            else
                counterIterationUpgrade = 0;
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


