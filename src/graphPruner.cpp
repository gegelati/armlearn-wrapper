/**
* \brief Executable for translating a .dot into a c file.
*/
#include <filesystem>

#include <gegelati.h>
#include "instructions.h"
#include "trainingParameters.h"
#include "armLearnLogger.h"

#include "ArmLearnWrapper.h"
#include "armLearningAgent.h"
int main(int argc, char** argv ){


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



    auto file = slashToAdd + trainingParams.testPath + "/out_best.dot";

    // Load graph
    std::cout << "Loading dot file from " << file << "." << std::endl;

    Environment dotEnv(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
    TPG::TPGGraph dotGraph(dotEnv, std::make_unique<TPG::TPGInstrumentedFactory>());
    File::TPGGraphDotImporter dot((file).c_str(), dotEnv, dotGraph);
    dot.importGraph();
    const TPG::TPGVertex* root = dotGraph.getRootVertices().front();



    if(!trainingParams.loadValidationTrajectories){
        armLearnEnv.updateValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
    } else {
        armLearnEnv.loadValidationTrajectories();
    }

    // Play the game once to identify useful edges & vertices
    std::ofstream ofs ("outLogs/tpg_orig.txt", std::ofstream::out);
    TPG::TPGExecutionEngineInstrumented tee(dotEnv);
    int nbActions = 0;
    int nbActionsEp = 0;
    int nbEpisodes = 0;
    double scoreOrig = 0;
    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            scoreOrig += armLearnEnv.getScore();
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            nbEpisodes++;
            nbActionsEp = 0;
        }
    	auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(* root).back()))->getActionID();
        armLearnEnv.doAction(actionID);
        ofs << nbActions << " " << actionID << std::endl;
        nbActions++;
        nbActionsEp++;
    }
    scoreOrig /= params.nbIterationsPerPolicyEvaluation;
    auto nbActionsOrig = nbActions;
    std::cout << "Total score: " << scoreOrig << " in "  << nbActionsOrig << " actions." << std::endl;
    ofs.close();

    // Clean the unused vertices & teams
    ((const TPG::TPGInstrumentedFactory&)dotGraph.getFactory()).clearUnusedTPGGraphElements(dotGraph);

    // Play the game again to check the result remains the same.
    std::ofstream ofs2 ("outLogs/tpg_clean.txt", std::ofstream::out);
    nbActions = 0;
    nbEpisodes = 0;
    double scoreClean = 0;
    armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
    std::cout << "Play with cleaned TPG code" << std::endl;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            scoreClean += armLearnEnv.getScore();
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            
            nbEpisodes++;
            nbActionsEp = 0;
        }
    	auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(* root).back()))->getActionID();
        armLearnEnv.doAction(actionID);
        ofs2 << nbActions << " " << actionID << std::endl;
        nbActions++;
        nbActionsEp++;
    }
    std::cout << "Total score: " << scoreClean / params.nbIterationsPerPolicyEvaluation << " in "  << nbActions << " actions." << std::endl;
    ofs.close();

    if(scoreClean / params.nbIterationsPerPolicyEvaluation != scoreOrig || nbActions != nbActionsOrig){
        std::cout << "Determinism was lost during graph cleaning." << std::endl;
        exit(1);
    }

    // Get stats on graph to get the required stack size
    std::cout << "Analyze graph." << std::endl;
    TPG::PolicyStats ps;
    ps.setEnvironment(dotEnv);
    ps.analyzePolicy(dotGraph.getRootVertices().front());

    // Print in file
    char bestPolicyStatsPath[150];
    std::ofstream bestStats;
    sprintf(bestPolicyStatsPath, "outLogs/out_best_stats_cleaned.md");
    bestStats.open(bestPolicyStatsPath);
    bestStats << ps;
    bestStats.close();

    // Export cleaned dot file
    std::cout << "Printing cleaned dot file." << std::endl;
    char bestDot[150];
    sprintf(bestDot, "outLogs/out_best_cleaned.dot");
    File::TPGGraphDotExporter dotExporter(bestDot, dotGraph);
    dotExporter.print();
    
    return 0;
}