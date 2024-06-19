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


void trainLearningAgent(
    Learn::ArmLearningAgent& la, ArmLearnWrapper& armLearnEnv, std::string pathConf, 
    uint64_t nbIterationTraining, uint64_t timeMaxTraining, uint64_t nbGenerations, bool useInternProgram
){

    // File for printing best policy stat.
    std::ofstream stats;
    stats.open((pathConf + "/bestPolicyStats.md").c_str());
    Log::LAPolicyStatsLogger logStats(la, stats);


    // Create an exporter for all graphs
    MARL::MarlTPGGraphDotExporter dotExporter((pathConf + "dotfiles/out_0000.dot").c_str(), *la.getTPGGraph(), useInternProgram);

    std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
    std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
    bool timeLimitReached = false;


    // Train for params.nbGenerations generations
    for (uint64_t i = 0; i < nbGenerations && !timeLimitReached; i++) {
        armLearnEnv.setgeneration(i);


        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(nbIterationTraining);

        std::ostringstream oss;
        oss << pathConf << "dotfiles/out_" << std::setfill('0') << std::setw(4) << i << ".dot";
        dotExporter.setNewFilePath(oss.str().c_str());
        dotExporter.print();

        la.trainOneGeneration(i);

        // Check time limit only if the parameter is above 0
        if(timeMaxTraining > 0){
            // Set true if the time is above the limit
            timeLimitReached = (((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count() > timeMaxTraining);
        }

    }
    stats.close();

}


std::vector<const TPG::TPGVertex *> selectSurvivingRoots(std::multimap<const TPG::TPGVertex *, std::multimap<double, bool>>& data, uint64_t nbPolicies){

    std::vector<const TPG::TPGVertex*> survivingRoots;
    std::vector<bool> successCleared;

    for(auto pair: data){
        for(auto pair2 : pair.second){
            std::cout<<" - "<<pair2.second;
        }std::cout<<std::endl;
    }

    for(uint64_t index = 0; index < nbPolicies; index++){

        // To select root with the best double fault error
        std::pair<const TPG::TPGVertex *, double> selectedRoot;
        std::vector<bool> successSelectedRoot;
        bool firstRoot = true;

        // For each root
        for(auto pair: data){

            // Init the success cleared to false
            if(successCleared.size() == 0){
                for(uint64_t indexInit = 0; indexInit < pair.second.size(); indexInit++){
                    successCleared.push_back(false);
                }
            }

            // Do not search for roots already selected
            if(std::find(survivingRoots.begin(), survivingRoots.end(), pair.first) == survivingRoots.end()){
            
                // Calcul the double fault error
                double doubleFaultError = 0;
                uint64_t i = 0;
                for(auto pairScoreSuccess: pair.second){
                    doubleFaultError += (pairScoreSuccess.second || successCleared[i]) ? 1 : 0;
                    i++;
                }
                
                if(firstRoot || doubleFaultError > selectedRoot.second){
                    firstRoot = false;
                    selectedRoot = std::make_pair(pair.first, doubleFaultError);

                    successSelectedRoot.clear();
                    for(auto pairScoreSuccess: pair.second){
                        successSelectedRoot.push_back(pairScoreSuccess.second);
                    }
                }
                std::cout<<doubleFaultError<<" - ";

            }
        }

        survivingRoots.push_back(selectedRoot.first);
        for(uint64_t i = 0; i < successCleared.size(); i++){
            successCleared[i] = (successCleared[i] || successSelectedRoot[i]);
        }std::cout<<std::endl;
    }



    return survivingRoots;
}

int main(int argc, char* argv[]) {
    std::cout << "Start ArmLearner application." << std::endl;

    std::string pathParams = "params/";
    if(argc > 1){
        pathParams = argv[1];
    }


    // This is important for the singularity image
    std::string slashToAdd = (std::filesystem::exists(("/" + pathParams + "/trainParams.json").c_str())) ? "/": "";
    std::cout<<"Status of slashToAdd : "<< slashToAdd<<std::endl;

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((slashToAdd + pathParams + "trainParams.json").c_str());


    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((slashToAdd + pathParams + "/params.json").c_str(), params);

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set, trainingParams);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

    armLearnEnv.loadTargetCSV(trainingParams.pathTargetCSV);

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
        armLearnEnv.saveValidationTrajectories(pathParams);
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories(pathParams);
    }

    std::multimap<const TPG::TPGVertex *, std::multimap<double, bool>> data;
    std::vector<std::shared_ptr<Learn::ArmLearningAgent>> listLa;

    int indexFile = 0;

    std::string path = (slashToAdd + trainingParams.pathLogs).c_str();

    for(int indexSeed = 0; indexSeed < trainingParams.federatedNbSeed; indexSeed++){

        // Generate files
        std::string pathConf = (path + "seed_" + std::to_string(indexSeed) + "/").c_str();
        std::filesystem::create_directory(pathConf);
        std::filesystem::create_directory(pathConf+ "dotfiles/");

        // Instantiate and init the learning agent
        listLa.push_back(std::make_shared<Learn::ArmLearningAgent>(armLearnEnv, set, params, trainingParams));
        std::shared_ptr<Learn::ArmLearningAgent> la = listLa.back();
        la->init(trainingParams.seed + indexSeed);

        //Creation of the Output stream on cout and on the file
        auto nameLogs = (!trainingParams.testing) ? "logsGegelati" : "garbage";
        std::ofstream fichier((pathConf + nameLogs + ".ods").c_str(), std::ios::out);
        auto logFile = *new Log::ArmLearnLogger(*la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion,fichier);
        auto logCout = *new Log::ArmLearnLogger(*la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion);

        trainLearningAgent(*la, armLearnEnv, pathConf, 
            trainingParams.nbIterationTraining, trainingParams.timeMaxTraining, 
            params.nbGenerations, params.mutation.marl.useInternProgram);

        // Update/Generate the training trajectories
        armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);

        // Keep the best policies
        auto bestRoots = la->keepBestPolicies(trainingParams.federatedNbPolicyKept);
        MARL::MarlTPGGraphDotExporter dotExporter((pathConf + "/out_best.dot").c_str(), *la->getTPGGraph(), params.mutation.marl.useInternProgram);
        dotExporter.print();

        std::cout<<1<<std::endl;
        // Generate the data of the policies
        auto seedData = la->generateDataOfRoots(bestRoots, armLearnEnv, params.nbIterationsPerPolicyEvaluation);

        std::cout<<2<<std::endl;
        data.insert(seedData.begin(), seedData.end());
        std::cout<<3<<std::endl;
    }

    // Sélection des nouvelles roots
    auto selectedRoots = selectSurvivingRoots(data, trainingParams.federatedNbPolicyChoose);

    // Generate files
    std::string pathConf = (path + "final/").c_str();
    std::filesystem::create_directory(pathConf);
    std::filesystem::create_directory(pathConf+ "dotfiles/");


    // Load params for the federated learning agent
    File::ParametersParser::loadParametersFromJson((slashToAdd + pathParams + "/federatedParams.json").c_str(), params);

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    la.init(trainingParams.seed);

    // Create new population
    la.createPopulationFromRoots(selectedRoots);

    //Creation of the Output stream on cout and on the file
    auto nameLogs = (!trainingParams.testing) ? "logsGegelati" : "garbage";
    std::ofstream fichier((pathConf + nameLogs + ".ods").c_str(), std::ios::out);
    auto logFile = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion,fichier);
    auto logCout = *new Log::ArmLearnLogger(la,trainingParams.doTrainingValidation,trainingParams.controlTrajectoriesDeletion);

    // Training
    trainLearningAgent(la, armLearnEnv, pathConf, 
                    trainingParams.nbIterationTraining, trainingParams.timeMaxTraining, 
                    params.nbGenerations, params.mutation.marl.useInternProgram);

    // Keep best policy
    la.keepBestPolicy();
    la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
    MARL::MarlTPGGraphDotExporter dotExporter((pathConf + "/out_best.dot").c_str(), *la.getTPGGraph(), params.mutation.marl.useInternProgram);
    dotExporter.print();
    
    // Export best policy statistics.
    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph()->getEnvironment());
    ps.analyzePolicy(la.getBestRoot().first);
    std::ofstream bestStats;
    bestStats.open((slashToAdd + "outLogs/out_best_stats.md").c_str());
    bestStats << ps;
    bestStats.close();

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    return 0;
}


