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


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson(ROOT_DIR "/trainParams.json");

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams.coefRewardNbIterations);

    // Prompt the number of threads
    std::cout << "Number of threads: " << params.nbThreads << std::endl;

    // If the training is progressive, set the limit to the param value
    if (trainingParams.progressiveModeTargets) armLearnEnv.setCurrentMaxLimitTarget(trainingParams.maxLengthTargets);

    // If the training is progressive, set the limit to the param value
    if (trainingParams.progressiveModeStartingPos) armLearnEnv.setCurrentMaxLimitStartingPos(trainingParams.maxLengthStartingPos);

    // Generate validation targets.
    if(params.doValidation){
        armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    }

    if(trainingParams.progressiveModeTargets){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(
            params.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);
    }


    // Instantiate and init the learning agent
    Learn::ParallelLearningAgent la(armLearnEnv, set, params);
    la.init();

#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif


    //Creation of the Output stream on cout and on the file
    std::ofstream fichier("logs.ods", std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,trainingParams.progressiveModeTargets,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,trainingParams.progressiveModeTargets);

    /*//Creation of CloudPoint.csv, point that the robotic arm ended to touch
    std::ofstream PointCloud;
    PointCloud.open("PointCloud.csv",std::ios::out);

    PointCloud << "Xp" << ";" << "Yp" << ";" << "Zp" << ";";
    PointCloud << "Xt" << ";" << "Yt" << ";" << "Zt" << ";" << "score" << std::endl; */

    // File for printing best policy stat.
    std::ofstream stats;
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger logStats(la, stats);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_0000.dot", *la.getTPGGraph());


    // Use previous Graphs
    if(trainingParams.startPreviousTPG){
        auto &tpg = *la.getTPGGraph();
        Environment env(set, armLearnEnv.getDataSources(), 8);
        File::TPGGraphDotImporter dotImporter((std::string(ROOT_DIR) + "/build/" + trainingParams.namePreviousTPG).c_str(), env, tpg);
    }

    // init Counter for upgrade the current max limit at 0
    int counterIterationUpgrade = 0;

    // Train for params.nbGenerations generations
    for (int i = 0; i < params.nbGenerations && !exitProgram; i++) {
        armLearnEnv.setgeneration(i);


        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(
            params.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);



	    //print the previous graphs
        char buff[16];
        sprintf(buff, "out_%04d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // we train the TPG, see doaction for the reward function
        la.trainOneGeneration(i);

        // Does a validation or not according to the parameter doValidation
        if (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos) {
            auto validationResults =
                la.evaluateAllRoots(i, Learn::LearningMode::TESTING);
            logFile.logAfterValidate(validationResults);
            logCout.logAfterValidate(validationResults);
        

            // log the current max limit
            logFile.logEnvironnementStatus(armLearnEnv.getCurrentMaxLimitTarget(), armLearnEnv.getCurrentMaxLimitStartingPos());
            logCout.logEnvironnementStatus(armLearnEnv.getCurrentMaxLimitTarget(), armLearnEnv.getCurrentMaxLimitStartingPos());
            
            // Get the best result of the training validation
            auto iter = validationResults.begin();
            std::advance(iter, validationResults.size() - 1);
            double bestResult = iter->first->getResult();

            // If the best TPG is above the threshold for upgrade
            if (bestResult > trainingParams.thresholdUpgrade){

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
                        params.nbIterationsPerPolicyEvaluation, trainingParams.doRandomStartingPosition, trainingParams.propTrajectoriesReused);
                }
            }
            // Reset the counter
            else
                counterIterationUpgrade = 0;
        }

        

    }


    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();


    // Export best policy statistics.
    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph()->getEnvironment());
    ps.analyzePolicy(la.getBestRoot().first);
    std::ofstream bestStats;
    bestStats.open("out_best_stats.md");
    bestStats << ps;
    bestStats.close();

    // close log file also
    stats.close();

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif

    return 0;
}


