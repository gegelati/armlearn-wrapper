#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <gegelati.h>

#include "ArmLearnWrapper.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 50
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

    std::vector<std::string> tparameters;
    std::string myparameter;
    std::ifstream myfile (ROOT_DIR"/TrainParam.txt");

    if ( myfile.is_open() ) {
        myfile >> myparameter;
        while(!myfile.eof()){
            std::for_each(myparameter.begin(), myparameter.end(), [](char & c) {
                c = ::tolower(c);
            });
            tparameters.push_back(myparameter);
            myfile >> myparameter;
        }
    }
    else{
        tparameters = {"3d","full","1","nostartingfile","noprogressive"};
    }
    /// [0] 2D/3D [1] close/large/full [2] Renew half/all targets (half not working) [3] StartingFile/NoStartingFile [3] Learning in a progressive space

    //Prototype to renew not all target
    float ar = std::stof(tparameters[2]); //We take a look a the parameters
    if(ar < 0 || ar > 1){ //If it is not in a possible value [0,1] (0 Mean no renew, 1 mean all, and if you are <0.5 the same value may pass on more than one generation)
        ar = 1; //We fixe it at the basic value
    }
    int NT = round(params.nbIterationsPerPolicyEvaluation*ar); //Otherwise, a calculate the true number of element to replace

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
    Learn::ParallelLearningAgent la(le, set, params);
    la.init();

#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false; // (set to false by other thread)
#endif


    //Adds a logger to the LA (to get statistics on learning) on std::cout et on a file
    //Creation of the name of the file
    std::string name="";
    for(auto & names : tparameters)
        name = name + names +"_";
    name.erase(name.end()-1,name.end());
    std::cout << name << std::endl;
    name = name + ".ods";

    //Creation of the Output stream on cout and on the file
    std::ofstream fichier(name, std::ios::out);
    auto logFile = *new Log::LABasicLogger(la,fichier);
    auto logCout = *new Log::LABasicLogger(la);

//    //Creation of CloudPoint.csv, point that the robotic arm ended to touch
//    std::ofstream PointCloud;
//    PointCloud.open("PointCloud.csv",std::ios::out);
//
//    PointCloud << "Xp" << ";" << "Yp" << ";" << "Zp" << ";";
//    PointCloud << "Xt" << ";" << "Yt" << ";" << "Zt" << ";" << "score" << std::endl;

    // File for printing best policy stat.
    std::ofstream stats;
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger logStats(la, stats);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_000.dot", *la.getTPGGraph());

    if(tparameters[3] == "startingfile"){
        auto &tpg = *la.getTPGGraph();
        Environment env(set, le.getDataSources(), 8);
        File::TPGGraphDotImporter dotImporter(ROOT_DIR"/cmake-build-release/out_006.dot", env, tpg);
    }



    // Train for NB_GENERATIONS generations
    for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++) {
        le.setgeneration(i);

        //Prototype to renew not all target
        std::for_each(le.trainingTargets.begin(), le.trainingTargets.begin()+NT, [](armlearn::Input<int16_t> * t){ delete t;}); //We delete the first part of target, to make a shift the value
        le.trainingTargets.erase(le.trainingTargets.begin(),le.trainingTargets.begin()+NT);

         for(int j=0;j<NT;j++){
            auto target = le.randomGoal(tparameters);
            le.trainingTargets.emplace_back(target);
        }


	    //print the previous graphs
        char buff[16];
        sprintf(buff, "out_%03d.dot", i);
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


