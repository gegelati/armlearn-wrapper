
#include <filesystem>
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
#include "armLearningAgent.h"


int main(int argc, char* argv[]){

    std::string pathParams = "params/";
    if(argc > 1){
        pathParams = argv[1];
    }


    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists(("/" + pathParams + "/trainParams.json").c_str())) ? "/": "";


    std::string repoConfig = (slashToAdd + pathParams + "repoConfig/").c_str();


    std::ifstream file((repoConfig + "launchMultiTraining.txt").c_str());
    int nbSeed;
    int nbTrainingConfig;
    if (file.is_open()) {
        // Lecture des deux nombres à partir du fichier
        file >> nbSeed >> nbTrainingConfig;

        // Fermeture du fichier
        file.close();
    }

    std::string pathLogs = (slashToAdd + "outLogs/").c_str();

    for(int indexConf = 0; indexConf < nbTrainingConfig; indexConf++){

        std::string pathConf = (pathLogs + "config_"+ std::to_string(indexConf) + "/").c_str();

        if(!std::filesystem::exists(pathConf)){
            std::filesystem::create_directory(pathConf);
            std::filesystem::create_directory((pathConf + "params/").c_str());
            std::filesystem::copy(repoConfig + "trainParams_" + std::to_string(indexConf) + ".json", (pathConf + "params/").c_str());
            std::filesystem::copy(repoConfig + "params_" + std::to_string(indexConf) + ".json", (pathConf + "params/").c_str());
        }


        for(int seed = 0; seed < nbSeed; seed++){

            // Create file with config and tout le tralala
            std::string path = (pathConf + "seed_" + std::to_string(seed) + "/").c_str();
            if(!std::filesystem::exists(path)){
                std::filesystem::create_directory(path);
                std::filesystem::create_directory((path + "dotfiles/").c_str());
            }

            std::cout<<"\nActually working with config " << indexConf << " and seed "<< seed<<"\n"<<std::endl;

            TrainingParameters trainingParams;
            trainingParams.loadParametersFromJson((pathConf + "params/trainParams_" + std::to_string(indexConf) + ".json").c_str());

            // Params of this config
            Learn::LearningParameters params;
            File::ParametersParser::loadParametersFromJson((pathConf + "params/params_" + std::to_string(indexConf) + ".json").c_str(), params);

            // Create the instruction set for programs
            Instructions::Set set;
            fillInstructionSet(set, trainingParams);

            // Instantiate the LearningEnvironment
            ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

            armLearnEnv.loadTargetCSV(trainingParams.pathTargetCSV);

            // Generate validation targets.
            if(params.doValidation){
                armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
                if(seed == 0 && indexConf == 0){
                    armLearnEnv.saveValidationTrajectories(pathParams);
                }
                armLearnEnv.loadValidationTrajectories(pathParams);
            }


            // Instantiate and init the learning agent
            Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);

            la.init(seed);

            //Creation of the Output stream on cout and on the file
            auto nameLogs = "logsGegelati";
            std::ofstream fichier((path + nameLogs + ".ods").c_str(), std::ios::out);
            auto logFile = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion,fichier);
            auto logCout = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion);

            // File for printing best policy stat.
            std::ofstream stats;
            stats.open((path + "bestPolicyStats.md").c_str());
            Log::LAPolicyStatsLogger logStats(la, stats);

            // Create an exporter for all graphs
            MARL::MarlTPGGraphDotExporter dotExporter((path + "dotfiles/out_0000.dot").c_str(), *la.getTPGGraph());

            std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
            std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
            bool timeLimitReached = false;


            // Train for params.nbGenerations generations
            for (uint64_t i = 0; i < params.nbGenerations && !timeLimitReached; i++) {
                armLearnEnv.setgeneration(i);

                // Update/Generate the training trajectories
                armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);

                //print the previous graphs
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


            auto &tpg = *la.getTPGGraph();
            Environment env(set, armLearnEnv.getDataSources(), 8);
            MARL::MarlTPGGraphDotImporter dotImporter((path + "out_best.dot").c_str(), env, tpg);
            trainingParams.testing = true;
            la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
            trainingParams.testing = false;

            // cleanup
            for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
                delete (&set.getInstruction(i));
            }
        }
    }

    return 0;
    
}