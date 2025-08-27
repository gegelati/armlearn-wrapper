#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>

#include <gegelati.h>
#include "instructions.h"
#include "trainingParameters.h"
#include "armLearnLogger.h"

#include "ArmLearnWrapper.h"
#include "armLearningAgent.h"


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

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson("params/trainParams.json");


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson("params/params.json", params);

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set, trainingParams);


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

    // Prompt the number of threads
    std::cout << "Number of threads: " << params.nbThreads << std::endl;

    // Generate validation targets.
    if(params.doValidation && !trainingParams.loadValidationTrajectories){
        armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    }


    if(trainingParams.doTrainingValidation){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    }


    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);

    la.init(trainingParams.seed);

    std::atomic<bool> exitProgram = false; // (set to false by other thread)
    std::thread threadKeyboard;

    if (trainingParams.interactiveMode && !trainingParams.testing){
#ifndef NO_CONSOLE_CONTROL

    threadKeyboard = std::thread(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif
    }



    // If a validation target is done
    bool doUpdateLimits = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);
    bool doValidationTarget = (trainingParams.doTrainingValidation && doUpdateLimits);

    //Creation of the Output stream on cout and on the file
    std::string nameLogs = (!!trainingParams.testing) ? "logsGegelati" : "garbage";
    std::ofstream fichier(("outLogs/" + nameLogs + ".ods"), std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion);



    // Use previous Graphs
    if(trainingParams.startPreviousTPG){
        auto &tpg = *la.getTPGGraph();
        Environment env = tpg.getEnvironment();
        File::TPGGraphDotImporter dotImporter(("outLogs/dotfiles/" + trainingParams.namePreviousTPG).c_str(), env, tpg);
    }

    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories();
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories();
    }

    if(trainingParams.testing){
        auto &tpg = *la.getTPGGraph();
        Environment env = tpg.getEnvironment();
        File::TPGGraphDotImporter dotImporter((trainingParams.testPath + "/out_best.dot").c_str(), env, tpg);
        la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
    } else {


        // File for printing best policy stat.
        std::ofstream stats;
        stats.open("outLogs/bestPolicyStats.md");
        Log::LAPolicyStatsLogger logStats(la, stats);

        // Create an exporter for all graphs
        File::TPGGraphDotExporter dotExporter("outLogs/dotfiles/out_0000.dot", *la.getTPGGraph());

        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
        std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
        bool timeLimitReached = false;

        // Train for params.nbGenerations generations
        for (uint64_t i = 0; i < params.nbGenerations && !exitProgram && !timeLimitReached; i++) {
            armLearnEnv.setgeneration(i);


            // Update/Generate the training trajectories
            armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);


            //print the previous graphs
            char buff[64];
            sprintf(buff,"outLogs/dotfiles/out_%04d.dot", static_cast<uint16_t>(i));
            dotExporter.setNewFilePath(buff);
            dotExporter.print();

            la.trainOneGeneration(i);

            // Check time limit only if the parameter is above 0
            if(trainingParams.timeMaxTraining > 0){
                // Set true if the time is above the limit
                timeLimitReached = (((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count() > trainingParams.timeMaxTraining);
            }

        }


        // Keep best policy
        la.keepBestPolicy();
        dotExporter.setNewFilePath("outLogs/out_best.dot");
        dotExporter.print();

        
        // Export best policy statistics.
        TPG::PolicyStats ps;
        ps.setEnvironment(la.getTPGGraph()->getEnvironment());
        ps.analyzePolicy(la.getBestRoot().first);
        std::ofstream bestStats;
        bestStats.open("outLogs/out_best_stats.md");
        bestStats << ps;
        bestStats.close();

        // close log file also
        stats.close();
    }



    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    if (trainingParams.interactiveMode && !trainingParams.testing) {
#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif
    }

    return 0;
}


