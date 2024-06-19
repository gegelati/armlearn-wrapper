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

int main(int argc, char* argv[]) {
    std::cout << "Start ArmLearner application." << std::endl;

    uint64_t seed = 0;
    if(argc > 1 && std::strcmp(argv[1], "default") != 0){
        seed = std::stoi(argv[1]);
    }

    std::string pathParams = "params/";
    if(argc > 2){
        pathParams = argv[2];
    }

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((pathParams + "trainParams.json").c_str());

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((pathParams + "/params.json").c_str(), params);

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set, trainingParams);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

    armLearnEnv.loadTargetCSV(trainingParams.pathTargetCSV, seed);

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
    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories(trainingParams.pathValidationTrajectories);
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories(trainingParams.pathValidationTrajectories);
    }
    std::string path = trainingParams.pathLogs;

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);

    la.init(seed);

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



    //Creation of the Output stream on cout and on the file
    auto nameLogs = (!trainingParams.testing) ? "logsGegelati" : "garbage";
    std::ofstream fichier((path + nameLogs + ".ods").c_str(), std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion);



    // Use previous Graphs
    if(trainingParams.startPreviousTPG){
        auto &tpg = *la.getTPGGraph();
        Environment env(set, armLearnEnv.getDataSources(), 8);
        MARL::MarlTPGGraphDotImporter dotImporter((path + "dotfiles/" + trainingParams.namePreviousTPG).c_str(), env, tpg);
    }

    if(trainingParams.testing){
        auto &tpg = *la.getTPGGraph();
        Environment env(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
        MARL::MarlTPGGraphDotImporter dotImporter((path + "out_best.dot").c_str(), env, tpg);
        la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
    } else {


        // File for printing best policy stat.
        std::ofstream stats;
        stats.open((path + "bestPolicyStats.md").c_str());
        Log::LAPolicyStatsLogger logStats(la, stats);

        // Create an exporter for all graphs
        MARL::MarlTPGGraphDotExporter dotExporter((path + "dotfiles/out_0000.dot").c_str(), *la.getTPGGraph(), params.mutation.marl.useInternProgram);

        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
        std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
        bool timeLimitReached = false;

        // Train for params.nbGenerations generations
        for (uint64_t i = 0; i < params.nbGenerations && !exitProgram && !timeLimitReached; i++) {
            armLearnEnv.setgeneration(i);


            // Update/Generate the training trajectories
            armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);


            std::ostringstream oss;
            oss << path << "dotfiles/out_" << std::setfill('0') << std::setw(4) << i << ".dot";
            dotExporter.setNewFilePath(oss.str().c_str());
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
        trainingParams.testing = true;
        la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
        dotExporter.setNewFilePath((path + "out_best.dot").c_str());
        dotExporter.print();

        
        // Export best policy statistics.
        TPG::PolicyStats ps;
        ps.setEnvironment(la.getTPGGraph()->getEnvironment());
        ps.analyzePolicy(la.getBestRoot().first);
        std::ofstream bestStats;
        bestStats.open((path + "out_best_stats.md").c_str());
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


