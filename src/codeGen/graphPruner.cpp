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

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((path + "params/trainParams.json").c_str());

    /*if(config < 11){
        trainingParams.useInstrSinLn=true;
    }*/

    // Set the parameters for the learning process.
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((path + "params/params.json").c_str(), params);

    // Create the instruction set for programs
	Instructions::Set set;
	fillInstructionSet(set, trainingParams);


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

    auto file = path + trainingParams.testPath;

    // Load graph
    std::cout << "Loading dot file from " << file << "." << std::endl;

    Environment dotEnv(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
    TPG::TPGGraph dotGraph(dotEnv, std::make_unique<TPG::TPGInstrumentedFactory>());
    File::TPGGraphDotImporter dot((file).c_str(), dotEnv, dotGraph);
    dot.importGraph();
    const TPG::TPGVertex* root = dotGraph.getRootVertices().front();



    armLearnEnv.loadValidationTrajectories();


    // Play the game once to identify useful edges & vertices
    std::ofstream ofs ((path + "outLogs/tpg_orig.txt").c_str(), std::ofstream::out);
    TPG::TPGExecutionEngineInstrumented tee(dotEnv);
    int nbActions = 0;
    int nbActionsEp = 0;
    int nbEpisodes = 0;
    double scoreOrig = 0;
    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        std::cout<<"ALLO"<<std::endl;
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            scoreOrig += armLearnEnv.getScore();
            armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
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
    dotGraph.clearProgramIntrons();


    root = dotGraph.getRootVertices().front();

    // Play the game again to check the result remains the same.
    std::ofstream ofs2 ((path + "outLogs/tpg_clean.txt").c_str(), std::ofstream::out);
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
    sprintf(bestPolicyStatsPath, (path + "outLogs/out_best_stats_cleaned.md").c_str());
    bestStats.open(bestPolicyStatsPath);
    bestStats << ps;
    bestStats.close();

    // Export cleaned dot file
    std::cout << "Printing cleaned dot file." << std::endl;
    char bestDot[150];
    sprintf(bestDot, (path + "outLogs/out_best_cleaned.dot").c_str());
    File::TPGGraphDotExporter dotExporter(bestDot, dotGraph);
    dotExporter.print();






    Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    la.init(trainingParams.seed);
    auto &tpg = *la.getTPGGraph();
    Environment env(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
    File::TPGGraphDotImporter dotImporter((path + "outLogs/out_best_cleaned.dot").c_str(), env, tpg);
    trainingParams.testPath = (path + "outLogs").c_str();
    trainingParams.testing = true;
    la.testingBestRoot(params.nbIterationsPerPolicyEvaluation);
    
    

    // Print graph
    std::string codeGenPath = (path + "outLogs/codeGen/").c_str();
    std::cout<<codeGenPath<<std::endl;
    if(!std::filesystem::exists(codeGenPath)){
        std::filesystem::create_directory(codeGenPath);
    }

    std::cout << "Printing C code." << std::endl;
	CodeGen::TPGGenerationEngineFactory factory(CodeGen::TPGGenerationEngineFactory::switchMode);
    std::unique_ptr<CodeGen::TPGGenerationEngine> tpggen = factory.create("codeGenArmlearn", dotGraph, codeGenPath);
    tpggen->generateTPGGraph();

    return 0;
}
