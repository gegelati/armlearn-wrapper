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
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams);

    // Prompt the number of threads
    std::cout << "Number of threads: " << params.nbThreads << std::endl;

    // Generate validation targets.
    if(params.doValidation){
        armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    }

    if(trainingParams.progressiveModeTargets){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    }


    // Instantiate and init the learning agent
    Learn::ParallelLearningAgent la(armLearnEnv, set, params);
    la.init(trainingParams.seed);

    std::atomic<bool> exitProgram = false; // (set to false by other thread)
    std::thread threadKeyboard;

    if (trainingParams.interactiveMode){
#ifndef NO_CONSOLE_CONTROL

    threadKeyboard = std::thread(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif
    }


    // If a validation target is done
    bool doValidationTarget = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);

    //Creation of the Output stream on cout and on the file
    std::ofstream fichier("logs.ods", std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,doValidationTarget,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,doValidationTarget);

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

    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories();
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories();
    }

    // Train for params.nbGenerations generations
    for (uint64_t i = 0; i < params.nbGenerations && !exitProgram; i++) {
        armLearnEnv.setgeneration(i);


        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(params.nbIterationsPerPolicyEvaluation);

	    //print the previous graphs
        char buff[16];
        sprintf(buff, "out_%04d.dot", static_cast<uint16_t>(i));
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // Log new generation
        logFile.logNewGeneration(i);
        logCout.logNewGeneration(i);

        // Populate Sequentially
        Mutator::TPGMutator::populateTPG(*la.getTPGGraph(), la.getArchive(),
                                        params.mutation, la.getRNG(),
                                        params.nbThreads);

        logFile.logAfterPopulateTPG();
        logCout.logAfterPopulateTPG();

        // Evaluate
        auto results = la.evaluateAllRoots(i, Learn::LearningMode::TRAINING);
        logFile.logAfterEvaluate(results);
        logCout.logAfterEvaluate(results);

        // Remove worst performing roots
        la.decimateWorstRoots(results);

        // Update the best
        la.updateEvaluationRecords(results);

        logFile.logAfterDecimate();
        logCout.logAfterDecimate();

        // Does a validation or not according to the parameter doValidation
        if (params.doValidation) {
            auto validationResults = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);
            logFile.logAfterValidate(validationResults);
            logCout.logAfterValidate(validationResults);
        }

        // Does a validation or not according to the parameter doValidation
        if (doValidationTarget) {
            auto trainingValidationResults = la.evaluateAllRoots(i, Learn::LearningMode::TESTING);
            logFile.logAfterTrainingValidate(trainingValidationResults);
            logCout.logAfterTrainingValidate(trainingValidationResults);
        

            // log the current max limit
            logFile.logEnvironnementStatus(armLearnEnv.getCurrentMaxLimitTarget(), armLearnEnv.getCurrentMaxLimitStartingPos());
            logCout.logEnvironnementStatus(armLearnEnv.getCurrentMaxLimitTarget(), armLearnEnv.getCurrentMaxLimitStartingPos());
            
            // Get the best result of the training validation
            auto iter = trainingValidationResults.begin();
            std::advance(iter, trainingValidationResults.size() - 1);
            double bestResult = iter->first->getResult();

            armLearnEnv.updateCurrentLimits(bestResult, params.nbIterationsPerPolicyEvaluation);
        }
        logFile.logEndOfTraining();
        logCout.logEndOfTraining();

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

    if (trainingParams.interactiveMode) {
#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif
    }

    return 0;
}


