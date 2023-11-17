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
    ArmLearnWrapper le;

    // Generate validation targets.
    le.validationTargets.clear();
    for(int j=0; j<params.nbIterationsPerPolicyEvaluation; j++){
        auto target = le.randomValidationGoal(tparameters);
        le.validationTargets.emplace_back(target);
    }


    // Generate first batch of training target, all of them will not be use. We do that in case we don't renew all target
    std::for_each(le.trainingTargets.begin(), le.trainingTargets.end(), [](armlearn::Input<int16_t> * t){ delete t;});
    le.trainingTargets.clear();

    // we create new targets
    for(int j=0; j<params.nbIterationsPerPolicyEvaluation; j++) {
        auto target = le.randomGoal(tparameters);
        le.trainingTargets.emplace_back(target);
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
    auto logFile = *new Log::ArmLearnLogger(la,fichier);
    auto logCout = *new Log::ArmLearnLogger(la);

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

    // Train for params.nbGenerations generations
    for (int i = 0; i < params.nbGenerations && !exitProgram; i++) {
        armLearnEnv.setgeneration(i);
        //Prototype to renew not all target
        std::for_each(le.trainingTargets.begin(), le.trainingTargets.begin()+NT, [](armlearn::Input<int16_t> * t){ delete t;}); //We delete the first part of target, to make a shift the value
        le.trainingTargets.erase(le.trainingTargets.begin(),le.trainingTargets.begin()+NT);

         for(int j=0;j<NT;j++){
            auto target = le.randomGoal(tparameters);
            le.trainingTargets.emplace_back(target);
        }

	    //print the previous graphs
        char buff[16];
        sprintf(buff, "out_%04d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // we train the TPG, see doaction for the reward function
        la.trainOneGeneration(i);
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


