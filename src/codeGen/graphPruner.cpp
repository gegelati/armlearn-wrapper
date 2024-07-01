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



int main(int argc, char* argv[]) {
    std::cout << "Start ArmLearner application." << std::endl;

    uint64_t seed = 0;
    if(argc > 1 && std::strcmp(argv[1], "default") != 0){
        seed = std::stoi(argv[1]);
    }

    std::string pathParams = "../params/";
    if(argc > 2){
        pathParams = argv[2];
    }



    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((pathParams + "/trainParams.json").c_str());

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }

    std::string pathOldGraph = "out_best.dot";
    if(argc > 4){
        pathOldGraph = argv[4];
    }

    std::string pathCleanedGraph = "out_best_cleaned.dot";
    if(argc > 5){
        pathCleanedGraph = argv[5];
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

    // Load graph
    std::cout << "Loading dot file from " << (trainingParams.pathLogs + pathOldGraph).c_str() << "." << std::endl;
    Environment dotEnv(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
    TPG::TPGGraph dotGraph(dotEnv, std::make_unique<TPG::TPGInstrumentedFactory>());
    File::TPGGraphDotImporter dot(((trainingParams.pathLogs + pathOldGraph).c_str()), dotEnv, dotGraph);
    dot.importGraph();

    armLearnEnv.loadValidationTrajectories(trainingParams.pathValidationTrajectories);
    TPG::TPGExecutionEngineInstrumented tee(dotEnv);

    double scoreOrig = 0;
    int nbActionsOrig = 0;
    int initNbRoots = dotGraph.getNbRootVertices();

    std::cout << "Play with TPG code" << std::endl;
    for(const TPG::TPGVertex * root: dotGraph.getRootVertices()){
        // Play the game once to identify useful edges & vertices
        int nbActions = 0;
        int nbActionsEp = 0;
        int nbEpisodes = 0;
        armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
        while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
            if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
                scoreOrig += armLearnEnv.getScore();
                armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
                nbEpisodes++;
                nbActionsEp = 0;
            }
            auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(*root).back()))->getActionID();
            std::vector<std::uint64_t> actionsID;
            actionsID.push_back(actionID);
            armLearnEnv.doActions(actionsID);
            nbActions++;
            nbActionsEp++;
        }
        nbActionsOrig += nbActions;
    }
    scoreOrig = scoreOrig / (double)(params.nbIterationsPerPolicyEvaluation * initNbRoots);
    
    std::cout << "Total score: " << scoreOrig << " in "  << nbActionsOrig << " actions." << std::endl;

    // Clean the unused vertices & teams
    ((const TPG::TPGInstrumentedFactory&)dotGraph.getFactory()).clearUnusedTPGGraphElements(dotGraph);
    dotGraph.clearProgramIntrons();


    double scoreClean = 0;
    int nbActionsAfter = 0;

    std::cout << "Play with cleaned TPG code" << std::endl;
    for(const TPG::TPGVertex * root: dotGraph.getRootVertices()){

        // Play the game again to check the result remains the same.
        int nbEpisodes = 0;
        int nbActions = 0;
        int nbActionsEp = 0;
        armLearnEnv.reset(0, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
        while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
            if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
                scoreClean += armLearnEnv.getScore();
                armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
                
                nbEpisodes++;
                nbActionsEp = 0;
            }
            auto actionID = ((TPG::TPGAction*)(tee.executeFromRoot(*root).back()))->getActionID();

            std::vector<std::uint64_t> actionsID;
            actionsID.push_back(actionID);
            armLearnEnv.doActions(actionsID);
            nbActions++;
            nbActionsEp++;

        }
        nbActionsAfter += nbActions;
    }
    scoreClean = scoreClean / (double)(params.nbIterationsPerPolicyEvaluation * initNbRoots);
    std::cout << "Total score: " << scoreClean << " in "  << nbActionsAfter << " actions." << std::endl;

    if(scoreClean != scoreOrig || nbActionsAfter != nbActionsOrig){
        std::cout << "Determinism was lost during graph cleaning." << std::endl;
        exit(1);
    }

    // Get stats on graph to get the required stack size
    std::cout << "Analyze graph." << std::endl;
    TPG::PolicyStats ps;
    ps.setEnvironment(dotEnv);
    ps.analyzePolicy(dotGraph.getRootVertices().front());

    // Print in file
    std::ofstream bestStats;
    std::ostringstream ossStats;
    ossStats << trainingParams.pathLogs << "out_best_stats_cleaned.md";
    bestStats.open(ossStats.str().c_str());
    bestStats << ps;
    bestStats.close();

    // Export cleaned dot file
    std::cout << "Printing cleaned dot file." << std::endl;
    std::ostringstream ossDot;
    ossDot << trainingParams.pathLogs << pathCleanedGraph;
    File::TPGGraphDotExporter dotExporter(ossDot.str().c_str(), dotGraph);
    dotExporter.print();




    /*Learn::ArmLearningAgent la(armLearnEnv, set, params, trainingParams);
    la.init(seed);
    auto &tpg = *la.getTPGGraph();
    Environment env(set, armLearnEnv.getDataSources(), params.nbRegisters, params.nbProgramConstant);
    File::TPGGraphDotImporter dotImporter((traububg + "outLogs/out_best_cleaned.dot").c_str(), env, tpg);
    trainingParams.pathLogs = (path + "outLogs").c_str();
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
    tpggen->generateTPGGraph();*/

    return 0;
}
