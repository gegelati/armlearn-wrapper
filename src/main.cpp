#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>

#include <gegelati.h>

#include "ArmLearnWrapper.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 20000
#endif

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
    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](double a, double b) -> double { return a - b; };
    auto add = [](double a, double b) -> double { return a + b; };
    auto times = [](double a, double b) -> double { return a * b; };
    auto divide = [](double a, double b) -> double { return a / b; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };
    auto cos = [](double a) -> double { return std::cos(a); };
    auto sin = [](double a) -> double { return std::sin(a); };

    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(times)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(divide)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond)));
    set.add(*(new Instructions::LambdaInstruction<double>(cos)));
    set.add(*(new Instructions::LambdaInstruction<double>(sin)));


    // Set the parameters for the learning process.
    // (Controls mutations probability, program lengths, and graph size
    // among other things)
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    std::vector<std::string> tparameters = {"2D","large","all"}; //Parameters for training
    /// [0] 2D/3D [1] close/large/(full not working currently) [2] Renew half/all targets (half not working)
    // Instantiate the LearningEnvironment
    ArmLearnWrapper le;

    // Generate validation targets.
    le.validationTargets.clear();
    for(int j=0; j<params.nbIterationsPerPolicyEvaluation; j++){
        auto target = le.randomGoal(tparameters);
        le.validationTargets.emplace_back(target);
    }

    // Instantiate and init the learning agent
    Learn::ParallelLearningAgent la(le, set, params);
    la.init();

#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif

    // Adds a logger to the LA (to get statistics on learning) on std::cout et on a file
        //Creation of the name of the file
    std::string name="";
    for(auto & names : tparameters)
        name = name + names +"_";
    name.erase(name.end()-1,name.end());
    name = name + ".ods";

        //Creation of the Output stream on cout and on the file
    std::ofstream fichier(name, std::ios::out);
    auto logFile = *new Log::LABasicLogger(la,fichier);
    auto logCout = *new Log::LABasicLogger(la);

    // File for printing best policy stat.
    std::ofstream stats;
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger logStats(la, stats);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_000.dot", la.getTPGGraph());
    /*auto &tpg = la.getTPGGraph();
    Environment env(set, le.getDataSources(), 8);
    File::TPGGraphDotImporter dotImporter(ROOT_DIR"/cmake-build-release/out_187.dot", env, tpg);

    dotImporter.importGraph();*/
/*
    // takes the first root of the graph, anyway out_best has only 1 root (the best)
    auto root = tpg.getRootVertices().front();*/

    // Train for NB_GENERATIONS generations
    for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++) {

        // we generate new random training targets so that at each generation, different targets are used.
        // we delete the old targets
        if(tparameters[2] == "all"){
            std::for_each(le.trainingTargets.end(), le.trainingTargets.end(), [](armlearn::Input<int16_t> * t){ delete t;});
            le.trainingTargets.clear();
        }

        // we create new targets
        if(tparameters[2] == "all"){
            for(int j=0; j<params.nbIterationsPerPolicyEvaluation; j++){
                auto target = le.randomGoal(tparameters);
                le.trainingTargets.emplace_back(target);
            }
        }


	    //print the previous graphs
        char buff[16];
        sprintf(buff, "out_%03d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // we train the TPG, see doaction for the reward fonction

        la.trainOneGeneration(i);

    }

    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();

    // Export best policy statistics.
    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph().getEnvironment());
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


