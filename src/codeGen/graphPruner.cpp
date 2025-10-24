/**
* \brief Executable for translating a .dot into a c file.
*/

extern "C" {

}

#include <filesystem>

#include <gegelati.h>
#include "../instructions.h"
#include "../trainingParameters.h"
#include "../armLearnLogger.h"

#include "../ArmLearnWrapper.h"
#include "../armLearningAgent.h"


int main(int argc, char** argv ){
    int config = 0;
    int seed = 0;

    std::string path = "";

    // Check if outLogs/CodeGen exists, if not, create it
    std::string codeGenPath = (path + "outLogs/CodeGen/").c_str();
    if(!std::filesystem::exists(codeGenPath)){
        std::filesystem::create_directories(codeGenPath);
    }

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((path + "params/trainParams.json").c_str());

    /*if(config < 11){
        trainingParams.useInstrSinLn=true;
    }*/

    // Set the parameters for the learning process.
    // Loads them from params.json
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((path + "params/params.json").c_str(), params);

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set, trainingParams);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

    auto dotfile = path + trainingParams.testPath;

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    la.init(trainingParams.seed);

    // Load graph
    std::cout << "Loading dot file from " << dotfile << std::endl;

    auto &tpg = *la.getTPGGraph();
    Environment env = tpg.getEnvironment();
    TPG::TPGGraph dotGraph(env, std::make_unique<TPG::TPGFactoryInstrumented>());
    File::TPGGraphDotImporter dot((dotfile).c_str(), env, dotGraph);
    dot.importGraph();
    const TPG::TPGVertex* root = dotGraph.getRootVertices().front();

    armLearnEnv.loadValidationTrajectories();

    // Play the game once to identify useful edges & vertices
    std::ofstream ofs ((path + "outLogs/tpg_orig.txt").c_str(), std::ofstream::out);
    TPG::TPGExecutionEngineInstrumented tee(env);
    int nbActions = 0;
    int nbActionsEp = 0;
    int nbEpisodes = 0;
    double scoreOrig = 0;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            scoreOrig += armLearnEnv.getScore();
            armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            nbEpisodes++;
            nbActionsEp = 0;
        }
    	auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(* root).first.back()))->getActionID();
        armLearnEnv.doAction(actionID);
        ofs << nbActions << " " << actionID << std::endl;
        nbActions++;
        nbActionsEp++;
    }
    scoreOrig /= params.nbIterationsPerPolicyEvaluation;
    auto nbActionsOrig = nbActions;
    std::cout << "Total score: " << scoreOrig << " in "  << nbActionsOrig << " actions." << std::endl;
    ofs.close();

    // Prune the unused vertices & teams
    ((const TPG::TPGFactoryInstrumented&)dotGraph.getFactory()).clearUnusedTPGGraphElements(dotGraph);
    dotGraph.clearProgramIntrons();

    root = dotGraph.getRootVertices().front();

    // Play the game again to check the result remains the same.
    std::ofstream ofs2 ((codeGenPath + "tpg_pruned.txt").c_str(), std::ofstream::out);
    nbActions = 0;
    nbEpisodes = 0;
    double scorePruned = 0;
    armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
    std::cout << "Play with pruned TPG code" << std::endl;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            scorePruned += armLearnEnv.getScore();
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            
            nbEpisodes++;
            nbActionsEp = 0;
        }
    	auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(* root).first.back()))->getActionID();

        armLearnEnv.doAction(actionID);
        ofs2 << nbActions << " " << actionID << std::endl;
        nbActions++;
        nbActionsEp++;

    }
    std::cout << "Total score: " << scorePruned / params.nbIterationsPerPolicyEvaluation << " in "  << nbActions << " actions." << std::endl;
    ofs.close();

    if(scorePruned / params.nbIterationsPerPolicyEvaluation != scoreOrig || nbActions != nbActionsOrig){
        std::cout << "Determinism was lost during graph pruning." << std::endl;
        exit(1);
    }

    // Get stats on graph to get the required stack size
    std::cout << "Analyze graph." << std::endl;
    TPG::PolicyStats ps;
    ps.setEnvironment(env);
    ps.analyzePolicy(dotGraph.getRootVertices().front());

    // Print in file
    char bestPolicyStatsPath[150];
    std::ofstream bestStats;
    sprintf(bestPolicyStatsPath, (codeGenPath + "best_root_pruned_stats.md").c_str());
    bestStats.open(bestPolicyStatsPath);
    bestStats << ps;
    bestStats.close();

    // Export pruned dot file
    std::cout << "Printing pruned dot file." << std::endl;
    char bestDot[150];
    sprintf(bestDot, (codeGenPath + "best_root_pruned.dot").c_str());
    File::TPGGraphDotExporter dotExporter(bestDot, dotGraph);
    dotExporter.print();

    File::TPGGraphDotImporter dotImporter((codeGenPath + "best_root_pruned.dot").c_str(), env, tpg);
    trainingParams.testPath = (path + "outLogs").c_str();
    trainingParams.testing = true;
    la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);

    std::cout << "Printing C code." << std::endl;
	CodeGen::TPGGenerationEngineFactory factory(CodeGen::TPGGenerationEngineFactory::switchMode);
    std::unique_ptr<CodeGen::TPGGenerationEngine> tpggen = factory.create("codeGenArmlearn", dotGraph, codeGenPath);
    tpggen->generateTPGGraph();

    return 0;
}
