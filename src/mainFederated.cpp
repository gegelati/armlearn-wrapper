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


std::vector<const TPG::TPGVertex *> selectSurvivingRoots(std::multimap<const TPG::TPGVertex *, std::vector<double>>& data, uint64_t nbPolicies){

    std::vector<const TPG::TPGVertex*> survivingRoots;
    std::vector<double> statusEvaliation;
    double oldDoubleFaultError = -std::numeric_limits<double>::infinity();

    for(auto pair: data){
        for(auto value : pair.second){
                std::cout<<";"<<value;
            
        }std::cout<<std::endl;
    }
    std::cout<<std::endl;


    for(uint64_t index = 0; index < nbPolicies; index++){

        // To select root with the best double fault error
        std::pair<const TPG::TPGVertex *, double> selectedRoot;
        std::vector<double> successSelectedRoot;
        bool firstRoot = true;

        // For each root
        for(auto pair: data){

            // Init the status of the evaluation to minus infinity
            if(statusEvaliation.size() == 0){
                for(uint64_t indexInit = 0; indexInit < pair.second.size(); indexInit++){
                    statusEvaliation.push_back(-std::numeric_limits<double>::infinity());
                }
            }

            // Do not search for roots already selected
            if(std::find(survivingRoots.begin(), survivingRoots.end(), pair.first) == survivingRoots.end()){
            
                // Calcul the double fault error
                double doubleFaultError = 0;
                uint64_t i = 0;
                for(auto value: pair.second){
                    doubleFaultError += std::max(value, statusEvaliation[i]);
                    i++;
                }
                
                if(firstRoot || doubleFaultError > selectedRoot.second){
                    firstRoot = false;
                    selectedRoot = std::make_pair(pair.first, doubleFaultError);

                    successSelectedRoot.clear();
                    for(auto value: pair.second){
                        successSelectedRoot.push_back(value);
                    }
                }
                std::cout<<doubleFaultError<<";";

            }
        }

        if(selectedRoot.second > oldDoubleFaultError){
            oldDoubleFaultError = selectedRoot.second;
            survivingRoots.push_back(selectedRoot.first);
            for(uint64_t i = 0; i < statusEvaliation.size(); i++){
                statusEvaliation[i] = std::max(statusEvaliation[i], successSelectedRoot[i]);
            }std::cout<<std::endl;
        } else {
            break;
        }


    }



    return survivingRoots;
}

int main(int argc, char* argv[]) {


    std::cout << "Do not Start Federation application." << std::endl;
    std::cout << "Deprecated for now." << std::endl;


    /*uint64_t seed = 0;
    if(argc > 1 && std::strcmp(argv[1], "default") != 0){
        seed = std::stoi(argv[1]);
    }

    uint64_t nbFederation = 1;
    if(argc > 2 && std::strcmp(argv[2], "default") != 0){
        nbFederation = std::stoi(argv[2]);
    }

    std::string pathParams = "../params/";
    if(argc > 3){
        pathParams = argv[3];
    }


    // This is important for the singularity image
    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((pathParams + "trainParams.json").c_str());

    if(argc > 4){
        trainingParams.pathLogs = argv[4];
    }

    std::string pathLoadData = "seed_/";
    if(argc > 5){
        pathLoadData = argv[5];
    }

    std::string pathSaveFederation = "federatedRun/";
    if(argc > 6){
        pathSaveFederation = argv[6];
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

    std::multimap<const TPG::TPGVertex *, std::vector<double>> data;
    std::vector<std::shared_ptr<Learn::ArmLearningAgent>> listLa;

    int indexFile = 0;

    std::string path = (trainingParams.pathLogs).c_str();

    // Update/Generate the training trajectories
    armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);

    for(int indexSeed = 0; indexSeed < nbFederation; indexSeed++){

        // Instantiate and init the learning agent
        listLa.push_back(std::make_shared<Learn::ArmLearningAgent>(armLearnEnv, set, params, trainingParams));
        std::shared_ptr<Learn::ArmLearningAgent> la = listLa.back();
        la->init(seed);



        // Find last generation
        std::ostringstream pathOutDot;
        pathOutDot << pathLoadData << indexSeed << "/dotfiles/out_lastGen.dot";

        auto &tpg = *la->getTPGGraph();

        std::cout<<"Graph load"<<std::endl;

        Environment env(set, armLearnEnv.getDataSources(), params.nbRegisters);
        File::TPGGraphDotImporter dotImporter(pathOutDot.str().c_str(), env, tpg);

        uint64_t nbPolicies = (1.0 - params.ratioDeletedRoots) * (double)params.mutation.tpg.nbRoots;

        std::vector<const TPG::TPGVertex *> bestRoots = la->keepBestPolicies(nbPolicies);

        // Generate the data of the policies
        auto seedData = la->generateDataOfRoots(bestRoots, armLearnEnv, params.nbIterationsPerPolicyEvaluation);
        std::cout<<"Data generated"<<std::endl;

        data.insert(seedData.begin(), seedData.end());
    }
    std::cout<<"Surviving roots selection"<<std::endl;

    // Sélection des nouvelles roots
    auto selectedRoots = selectSurvivingRoots(data, trainingParams.nbRootsKept);

    std::cout<<"Initialisation of new learning agent"<<std::endl;
    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    la.init(seed);



    std::cout<<"Creating population from roots"<<std::endl;
    // Create new population
    la.createPopulationFromRoots(selectedRoots);

    //if(trainingParams.dispableDuplication)
    //la.disableNonRootsVertexDuplication();

    std::cout<<"Saving dot file"<<std::endl;
    std::cout<<pathSaveFederation<<std::endl;
    File::TPGGraphDotExporter dotExporter((pathSaveFederation + "/dotfiles/out_0000.dot").c_str(), *la.getTPGGraph());
    std::ostringstream oss;
    oss << pathSaveFederation << "/dotfiles/out_0000.dot";
    dotExporter.setNewFilePath(oss.str().c_str());
    dotExporter.print();

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }*/

    return 0;
}


