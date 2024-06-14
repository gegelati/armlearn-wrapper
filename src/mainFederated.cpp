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



    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((slashToAdd + "params/trainParams.json").c_str());


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((slashToAdd + "params/params.json").c_str(), params);

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

    // If a validation target is done
    bool doUpdateLimits = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);
    bool doValidationTarget = (trainingParams.doTrainingValidation && doUpdateLimits);


    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories();
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories();
    }

    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> results;


    int nbSeeds = 5;
    int nbPoliciesKept = 2;
    int indexFile = 0;
    while(std::filesystem::exists((slashToAdd + "outLogs/federated_"+ std::to_string(indexFile) + "/").c_str())){
        indexFile++;
    }
    std::string path = (slashToAdd + "outLogs/federated_"+ std::to_string(indexFile) + "/").c_str();
    std::filesystem::create_directory(path);

    for(int indexSeed = 0; indexSeed < nbSeeds; indexSeed++){

        // Generate files
        std::string pathConf = (path + "seed_" + std::to_string(indexSeed) + "/").c_str();
        std::filesystem::create_directory(pathConf);
        std::filesystem::create_directory(pathConf+ "dotfiles/");

        // Instantiate and init the learning agent
        Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
        la.init(trainingParams.seed);




        //Creation of the Output stream on cout and on the file
        auto nameLogs = (!trainingParams.testing) ? "logsGegelati" : "garbage";
        std::ofstream fichier((pathConf + nameLogs + ".ods").c_str(), std::ios::out);
        auto logFile = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion,fichier);
        auto logCout = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion);


        
        // File for printing best policy stat.
        std::ofstream stats;
        stats.open((pathConf + "/bestPolicyStats.md").c_str());
        Log::LAPolicyStatsLogger logStats(la, stats);


        // Create an exporter for all graphs
        MARL::MarlTPGGraphDotExporter dotExporter((pathConf + "dotfiles/out_0000.dot").c_str(), *la.getTPGGraph(), params.mutation.marl.useInternProgram);

        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
        std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
        bool timeLimitReached = false;


        // Train for params.nbGenerations generations
        for (uint64_t i = 0; i < params.nbGenerations && !timeLimitReached; i++) {
            armLearnEnv.setgeneration(i);


            // Update/Generate the training trajectories
            armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);


            //print the previous graphs
            char buff[16];
            sprintf(buff, (pathConf + "dotfiles/out_%04d.dot").c_str(), static_cast<uint16_t>(i));
            dotExporter.setNewFilePath(buff);
            dotExporter.print();

            la.trainOneGeneration(i);

            // Check time limit only if the parameter is above 0
            if(trainingParams.timeMaxTraining > 0){
                // Set true if the time is above the limit
                timeLimitReached = (((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count() > trainingParams.timeMaxTraining);
            }

        }
        
        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);
        auto seedResults = la.keepBestPolicies(nbPoliciesKept);
        dotExporter.setNewFilePath((pathConf + "/out_best.dot").c_str());
        dotExporter.print();


        results.insert(seedResults.begin(), seedResults.end());
    }

    std::cout<<0<<std::endl;
    results.end()->second->getOutgoingEdges().back()->getDestination();
    std::cout<<1<<std::endl;
    

    // Etude des résultats

    // Sélection des nouvelles roots

    // Création de la nouvelle population

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    std::cout<<2<<std::endl;
    results.rbegin()->second->getOutgoingEdges().back()->getDestination();
    la.init(trainingParams.seed);
    std::cout<<3<<std::endl;
    results.rbegin()->second->getOutgoingEdges().back()->getDestination();
    std::cout<<4<<std::endl;
    la.createPopulationFromResults(results);




    // Entrainement

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    return 0;
}


