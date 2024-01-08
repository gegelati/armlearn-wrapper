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

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set);

    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((slashToAdd + "params/trainParams.json").c_str());


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((slashToAdd + "params/params.json").c_str(), params);



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

    // If a validation target is done
    bool doUpdateLimits = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);
    bool doValidationTarget = (trainingParams.doTrainingValidation && doUpdateLimits);

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, doValidationTarget, doUpdateLimits);

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



    //Creation of the Output stream on cout and on the file
    std::ofstream fichier((slashToAdd + "outLogs/logs.ods").c_str(), std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits);

    /*//Creation of CloudPoint.csv, point that the robotic arm ended to touch
    std::ofstream PointCloud;
    PointCloud.open("PointCloud.csv",std::ios::out);

    PointCloud << "Xp" << ";" << "Yp" << ";" << "Zp" << ";";
    PointCloud << "Xt" << ";" << "Yt" << ";" << "Zt" << ";" << "score" << std::endl; */

    // File for printing best policy stat.
    std::ofstream stats;
    stats.open((slashToAdd + "outLogs/bestPolicyStats.md").c_str());
    Log::LAPolicyStatsLogger logStats(la, stats);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter((slashToAdd + "outLogs/out_0000.dot").c_str(), *la.getTPGGraph());


    // Use previous Graphs
    if(trainingParams.startPreviousTPG){
        auto &tpg = *la.getTPGGraph();
        Environment env(set, armLearnEnv.getDataSources(), 8);
        File::TPGGraphDotImporter dotImporter((slashToAdd + "outLogs/" + trainingParams.namePreviousTPG).c_str(), env, tpg);
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
        sprintf(buff, (slashToAdd + "outLogs/out_%04d.dot").c_str(), static_cast<uint16_t>(i));
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        la.trainOneGeneration(i);

    }


    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath((slashToAdd + "outLogs/out_best.dot").c_str());
    dotExporter.print();


    // Export best policy statistics.
    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph()->getEnvironment());
    ps.analyzePolicy(la.getBestRoot().first);
    std::ofstream bestStats;
    bestStats.open((slashToAdd + "outLogs/out_best_stats.md").c_str());
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


