
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


int main(){




    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";


    std::string repoConfig = (slashToAdd + "params/repoConfig/").c_str();

    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((repoConfig + "params.json").c_str(), params);


    std::ifstream file((repoConfig + "launchMultiTraining.txt").c_str());
    int nbSeed;
    int nbTrainingConfig;
    if (file.is_open()) {
        // Lecture des deux nombres à partir du fichier
        file >> nbSeed >> nbTrainingConfig;

        // Fermeture du fichier
        file.close();
    }

    for(int indexConf = 0; indexConf < nbTrainingConfig; indexConf++){

        

        for(int seed = 0; seed < nbSeed; seed++){

            // Create file with config and tout le tralala
            std::string path = (slashToAdd + "outLogs/config_"+ std::to_string(indexConf) + "_" + std::to_string(seed) + "/").c_str();
            if(!std::filesystem::exists(path)){
                std::filesystem::create_directory(path);
                std::filesystem::create_directory((path + "outLogs/").c_str());
                std::filesystem::create_directory((path + "outLogs/dotfiles/").c_str());
                std::filesystem::create_directory((path + "params/").c_str());

                std::filesystem::copy(repoConfig + "trainParams_" + std::to_string(indexConf) + ".json", (path + "params/").c_str());
                std::filesystem::copy(repoConfig + "params.json", (path + "params/").c_str());
            }



            TrainingParameters trainingParams;
            trainingParams.loadParametersFromJson((path + "params/trainParams_" + std::to_string(indexConf) + ".json").c_str());

            // Create the instruction set for programs
            Instructions::Set set;
            fillInstructionSet(set, trainingParams);

            // Instantiate the LearningEnvironment
            ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

            // Generate validation targets.
            if(params.doValidation){
                armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
                if(seed == 0 && indexConf == 0){
                    armLearnEnv.saveValidationTrajectories();
                }
                armLearnEnv.loadValidationTrajectories();
            }

            if(trainingParams.progressiveModeTargets){
                // Update/Generate the first training validation trajectories
                armLearnEnv.updateTrainingValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
            }


            // Instantiate and init the learning agent
            Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);

            la.init(seed);

            // If a validation target is done
            bool doUpdateLimits = (trainingParams.progressiveModeTargets || trainingParams.progressiveModeStartingPos);
            bool doValidationTarget = (trainingParams.doTrainingValidation && doUpdateLimits);

            //Creation of the Output stream on cout and on the file
            auto nameLogs = "logsGegelati";
            std::ofstream fichier((path + "outLogs/" + nameLogs + ".ods").c_str(), std::ios::out);
            auto logFile = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion,fichier);
            auto logCout = *new Log::ArmLearnLogger(la,doValidationTarget,doUpdateLimits,trainingParams.controlTrajectoriesDeletion);

            // File for printing best policy stat.
            std::ofstream stats;
            stats.open((path + "outLogs/bestPolicyStats.md").c_str());
            Log::LAPolicyStatsLogger logStats(la, stats);

            // Create an exporter for all graphs
            File::TPGGraphDotExporter dotExporter((path + "outLogs/dotfiles/out_0000.dot").c_str(), *la.getTPGGraph());

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
                sprintf(buff, (path + "outLogs/dotfiles/out_%04d.dot").c_str(), static_cast<uint16_t>(i));
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
            dotExporter.setNewFilePath((path + "outLogs/out_best.dot").c_str());
            dotExporter.print();

            
            // Export best policy statistics.
            TPG::PolicyStats ps;
            ps.setEnvironment(la.getTPGGraph()->getEnvironment());
            ps.analyzePolicy(la.getBestRoot().first);
            std::ofstream bestStats;
            bestStats.open((path + "outLogs/out_best_stats.md").c_str());
            bestStats << ps;
            bestStats.close();


            // close log file also
            stats.close();

            auto &tpg = *la.getTPGGraph();
            Environment env(set, armLearnEnv.getDataSources(), 8);
            File::TPGGraphDotImporter dotImporter((path + "outLogs/out_best.dot").c_str(), env, tpg);
            trainingParams.testPath = (path + "outLogs").c_str();
            trainingParams.testing = true;
            la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);

            // cleanup
            for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
                delete (&set.getInstruction(i));
            }
        }
    }

    return 0;
    
}