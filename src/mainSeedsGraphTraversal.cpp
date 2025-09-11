#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <gegelati.h>
#include <random>
#include <vector>
#include<unistd.h>

#include "instructions.h"
#include "trainingParameters.h"
#include "armLearnLogger.h"
#include "ArmLearnWrapper.h"
#include "armLearningAgent.h"

#define DEFAULT_NB_SEEDS_TO_SEARCH 2E1 // number of seeds used to find graph traversals
#define NB_VALUES_PER_CLASS 25 // number of occurences of each graph traversal we want to have
// #define VERBOSE

/* Pourquoi générer des seeds pour la traversée de graphe ?

Dans le cadre des TPGs, une seed définit une position de l’environnement d’apprentissage utilisée pour l’évaluation. 
Le script de génération de seeds sert à équilibrer les parcours de graphe, c’est-à-dire à garantir que chaque trajet 
réalisable dans le graphe soit représenté le même nombre de fois.

Lors de l’inférence, cependant, les TPGs manifestent une préférence pour certaines actions, et donc pour certains parcours. 
Cela reflète leur capacité à s’adapter aux situations rencontrées dans l’environnement (par exemple sortir d’un blocage 
contre un mur ou stabiliser un pendule inversé). Autrement dit, le graphe du TPG n’explore pas ses parcours de manière 
équilibrée par défaut.

Ce biais n’affecte pas directement la qualité de la réponse du TPG face à son environnement d’apprentissage. En revanche, 
il complique l’évaluation des performances. En effet, si l’on se contente d’observer les parcours réellement suivis par 
le TPG à partir de quelques positions tirées aléatoirement dans l’environnement, certaines actions/parcours seront 
sous-représentés.

Pour obtenir des statistiques fiables, il est donc nécessaire de mesurer chaque type de parcours le même nombre de fois, 
y compris ceux qui seraient rares lors d’une exécution normale en inférence.
*/

/// @brief this code generates and stores the parameters required at the start
/// of an ensemble of graph traversal measurements.
/// Graph traversal measurements are use to compute metrics 

/// The storing of traceTeamIds, which represents the
/// path of the graph traversal is stored for indicative
/// purpose.

/// specify if the seeds are randomized in the output header file
/// allows to distribute the computation and be less dependent to the heating
/// of the chip.
bool randomizeSeeds = true;

/// @brief Function to write the content of inferenceTraceInfos to a C Header file called
/// seeds_nbActoinsToTerminal.h, used to write starting position of the angle, velocity of the
/// Learning Environment 
void storeToHeaderFile(
    const std::string &filename,
    const std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>> mapITI,
    size_t nbDataSources,
    bool randomize);

/// @brief Function to extract all doubles from a DataHandler
/// @param handler the DataHandler to extract doubles from
/// @return a vector of doubles extracted from the DataHandler
std::vector<double> extractAllDoubles(const Data::DataHandler& handler);


int main(int argc, char *argv[])
{

    std::cout << "\033[1;33m=====[ Generate Seeds for Graph Traversal target ]=====\033[0m" << std::endl;


    /*
     * This program needs 2 arguments :
     * - argv[1] : path to the .dot file of the TPG to be used, if relative path, starts from call pwd
     * - argv[2] : seed
     */

    /* Checking arguments */

    if (argc < 3)
    {
        std::cerr << "Missing arguments, this program needs (in order) : the path to the .dot file, a seed." << std::endl;
        exit(1);
    }

    std::filesystem::path dotPath(argv[1]);
    unsigned int initial_seed_RNG;
    int nbSeedsToSearch = DEFAULT_NB_SEEDS_TO_SEARCH;

    try
    {
        initial_seed_RNG = (unsigned int)std::stoi(argv[2]);
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "seed is not an int" << std::endl;
        exit(1);
    }

    // Set the seed for the RNG
    srand(initial_seed_RNG);

    /* Settings */

    // Load parameters
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson("params/params.json", params);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson("params/trainParams.json");

    // Instruction set
    Instructions::Set set;
    fillInstructionSet(set, trainingParams);

    /* Setup armLearn (simulator of robot arm control) Learning Environment environment and import graph */

    // Setup Learning Environment 
    /// True if gegelati (TPG) is running in the LE, false if its an other algorithm, which would be SAC for instance
    bool gegelatiRunning = true;
    ArmLearnWrapper armLE(params.maxNbActionsPerEval, trainingParams, gegelatiRunning);
     // Calculer le nombre total de dataSources (somme des dimensions de chaque DataHandler)
    size_t nbDataSources = 0;
    auto dataHandlers = armLE.getDataSources();
    for (const auto& handlerRef : dataHandlers) {
        nbDataSources += handlerRef.get().getDimensionsSize().at(0);
    }
    std::cout << "Number of data sources in the Learning Environment: " << nbDataSources << std::endl;

    // Instantiate and init the learning agent
    Learn::ArmLearningAgent la(armLE, set, params, trainingParams);
    la.init(trainingParams.seed);

    // Load Envrionment (needed to execute a program)


    // Initialisation par le constructeur
    Environment env(set, params, armLE.getDataSources());
    
    // Load graph from dot file

    TPG::TPGGraph tpgGraph(env, std::make_unique<TPG::TPGFactoryInstrumented>());
    File::TPGGraphDotImporter tpgGraphDotImporter(dotPath.c_str(), env, tpgGraph);
    tpgGraphDotImporter.importGraph();

    /* Prepare for inference, retrieve root and execution engine */

    // B. Jusqu'ici, on récupérait la racine la plus ancienne (la première créée) donc back() 
    const TPG::TPGVertex *root(tpgGraph.getRootVertices().back()); //first.back()

    // Retrieve execution engine
    TPG::TPGExecutionEngineInstrumented tee(env);


    /* Prepare to retrieve graph traversal informations */

    // Prepare for Execution informations extraction and export
    TPG::ExecutionInfos executionInfos;

    // Annotate the graph to understand the progress of the execution
    executionInfos.assignIdentifiers((const TPG::TPGTeamInstrumented *)root);

    std::cout << "completed assignIdentifiers()"<< std::endl;

    // We don't know the size of the following struct at compile time
    // since we want NB_VALUES_PER_CLASS occurences of each graph traversals and we don't
    // know how many graph traversal there can be for a given TPG a priori
    // Graph traversal = combinations of Team traversed before reaching action
    std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>> mapITI;

    /* TPG Inference */

    int continue_search = 0;
    int total_nb_seeds = 0;
    int max_nb_seeds_to_search = 2E1; // to avoid infinite loop

    do
    {
        std::cout << "\n\033[1;33m----- New serie of search through " << nbSeedsToSearch << " seeds -----\033[0m" << std::endl;

        // allow the generation of training validation trajectories which are test trajectories 
        trainingParams.doTrainingValidation = true; 
        params.nbIterationsPerPolicyEvaluation = nbSeedsToSearch; // we want to generate nbSeedsToSearch trajectories
        
        if(trainingParams.doTrainingValidation){
            // Update/Generate the first training validation trajectories
            armLE.updateTrainingValidationTrajectories(params.nbIterationsPerPolicyEvaluation);
        }

        std::cout << "completed updateTrainingValidationTrajectories()"<< std::endl;

        for (int j = 0; j < nbSeedsToSearch; j++)
        {
            // generate the seed, it is used to find an initial value for each dataSource 
            // of the LearningEnvironment
        
            // std::cout << "searching seed: " << j << std::endl;

            // set the inital arm Learn Wrapper conditions using the seed
            armLE.reset(j, Learn::LearningMode::TESTING, j, 0);

            // std::cout << "reseted seed: " << j << std::endl;

            // get the data sources from the LE after reset to store them in the inferenceTraceInfos   
            std::vector<std::reference_wrapper<const Data::DataHandler>> dataHandlers = armLE.getDataSources();
            
            // extract all doubles from all DataHandlers
            std::vector<double> dataSourcesLE;
            for (const auto& handlerRef : dataHandlers) {
                const Data::DataHandler& handler = handlerRef.get();
                std::vector<double> extracted = extractAllDoubles(handler);
                dataSourcesLE.insert(dataSourcesLE.end(), extracted.begin(), extracted.end());
            }
            if (j==0){  //afficher les dataSourcesLE
            std::cout << "dataSourcesLE: ";
            for (double d : dataSourcesLE) {
                std::cout << d << " ";
            }
            std::cout << std::endl;}
          

            // execute one action, trace it, and move to the next seed.
            tee.executeFromRoot(*root);

            // std::cout << "executed seed: " << j << std::endl;

            // retrieve graph traversal informations from TPG and tee
            executionInfos.analyzeExecution(tee, tpgGraph, j, dataSourcesLE);
        }

        // ended a serie of search through nbSeedsToSearch

        // retrieve vecInfTraceInfos
        std::vector<TPG::InferenceTraceInfos> vecInferenceTraceInfos = executionInfos.getVecInferenceTraceInfos();

        for (TPG::InferenceTraceInfos infTraceInfos : vecInferenceTraceInfos)
        {
            // the key is the graph traversal of the inferenceTraceInfos object under inspection
            // graph Traversals  = list<int> traceTeamIds

            // if the key doesnt exist in the map, insert it with its value
            if (!mapITI.count(infTraceInfos.traceTeamIds))
            {
                //insert the infTraceInfos
                std::vector<TPG::InferenceTraceInfos> vecITI = {infTraceInfos};
                // insert key (graphTraversal), value (iTI)
                mapITI.insert({infTraceInfos.traceTeamIds, vecITI});
            }

            // else if the key is already present but we have less than NB_VALUES_PER_CLASS values and the seed is not already in the map,
            // insert the value
            else
            {
                if (mapITI[infTraceInfos.traceTeamIds].size() < NB_VALUES_PER_CLASS)
                {

                    // iterate over the vector mapITI[infTraceInfos.traceTeamIds] to make sure the seed
                    // we want to add is not already present
                    int collision = 0;
                    for (TPG::InferenceTraceInfos iTI : mapITI[infTraceInfos.traceTeamIds])
                    {
                        if (infTraceInfos.seed == iTI.seed)
                        {
                            collision++;
                        }
                    }
                    if (!collision)
                    {
                        // insert infTraceInfos in pre-existing vector at key infTraceInfos.traceTeamIds of the map mapITI
                        mapITI[infTraceInfos.traceTeamIds].push_back(infTraceInfos);
                    }
                    else
                    {
                        std::cerr << "collision" << std::endl;
                    }
                }
            }

            // else discard key, value pair
        }

        // Display status of the map of inferenceTraceInfos
        std::cout << "\nStatus of mapITI after this round:\n";
        for (const auto& [traceTeams, vecITI] : mapITI) {
            std::cout << "[";
            bool first = true;
            for (int t : traceTeams) {
                if (!first) std::cout << " -> ";
                std::cout << "T" << t;
                first = false;
            }
            std::cout << "] : " << vecITI.size() << " / " << NB_VALUES_PER_CLASS << std::endl;
        }
        std::cout << "Total traversals: " << mapITI.size() << std::endl;


        // Do we have a balanced map ? i.e the same number of values for each graph traversal (NB_VALUES_PER_CLASS)
        // if yes, stop searching
        // if not, continue searching, meaning go over new seeds, and exceed nbSeedsToSearch.
        int balanced = 1;
        for (std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>>::iterator it = mapITI.begin(); it != mapITI.end(); it++)
        {
            const std::vector<TPG::InferenceTraceInfos> &iti = it->second;
            if (iti.size() < NB_VALUES_PER_CLASS)
            {
                balanced = 0;
            }
        }

        // if not balanced -> continue_serach 
        // but we need to limit the time spent in this loop at some point
        // so we stop if we reach max_nb_seeds_to_search
        // When the Learning Environment is more complex, the size of the TPG is larger
        // and the number of graph traversals can be very large. Additionnaly, the time to 
        // compute one inference and to reset the LE is also larger.
        // So we need to to bound this loop to avoid spending days in it.
        // The value of max_nb_seeds_to_search can be increased if the user wants to spend more
        // time in this loop to try to get a more balanced map.
        continue_search = !balanced && (total_nb_seeds < max_nb_seeds_to_search);
        total_nb_seeds += nbSeedsToSearch;

        // clear vecInferenceTraceInfos
        executionInfos.clear();

    } while (continue_search);

    std::cout << "total seeds searched: " << total_nb_seeds << std::endl;
    std::cout << "graph traversal: " << mapITI.size() << std::endl;
    for (std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>>::iterator it = mapITI.begin(); it != mapITI.end(); it++)
    {
        const std::list<int> &teams = it->first;
        const std::vector<TPG::InferenceTraceInfos> &iti = it->second;

        std::cout << "[";
        for (int t : teams)
        {
            std::cout << "T" << t << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << iti.size() << "\n";
    }

    // Write data to CSV file
    storeToHeaderFile("outLogs/PreCalcul/seeds_nbActionsToTerminal.h", mapITI, nbDataSources, randomizeSeeds);

    // Empty the vec of InferenceTraceInfos from executionInfos which has current TPG execution context in it
    executionInfos.clear();

    // Re-fill it with the values from the map we are keeping
    std::vector<TPG::InferenceTraceInfos> overallInfTraceInfos; 
    for (std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>>::iterator it = mapITI.begin(); it != mapITI.end(); it++)
    {
        const std::vector<TPG::InferenceTraceInfos>& iTI = it->second;
        copy(iTI.begin(), iTI.end(), back_inserter(overallInfTraceInfos));
    }
    
    executionInfos.setVecInferenceTraceInfos(overallInfTraceInfos);
    executionInfos.writeTPGtoJson("outLogs/PreCalcul/tpgInfos.json");
    executionInfos.writeInfosToJson("outLogs/PreCalcul/executionInfos.json");

    std::cout << "End program" << std::endl;

    return 0;
}

std::vector<double> extractAllDoubles(const Data::DataHandler& handler)
{
    std::vector<double> result;

    // Combien de double sont stockés ?
    size_t n = handler.getAddressSpace(typeid(const double));

    result.reserve(n);
    for (size_t i = 0; i < n; i++) {
        double value = *handler.getDataAt(typeid(const double), i).getSharedPointer<const double>();
        result.push_back(value);
    }

    return result;
}


void storeToHeaderFile(
    const std::string &filename,
    const std::map<std::list<int>, std::vector<TPG::InferenceTraceInfos>> mapITI,
    size_t nbDataSources,
    bool randomize)
{

    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Collect all data into vectors
    std::vector<std::vector<double>> dataSources; // [nbValues][NB_DATA_SOURCES]
    std::vector<unsigned int> seeds;
    std::vector<unsigned int> ids_graph_traversals;

    // Write traversal mapping as comments
    file << "// ===== Graph Traversal Mapping =====\n";
    int id_GT = 0;
    for (const auto &[traceTeams, vecITI] : mapITI)
    {
        file << "// [" << id_GT << "] -> [";
        bool first = true;
        for (int t : traceTeams) {
            if (!first) file << " -> ";
            file << "T" << t;
            first = false;
        }
        file << "]\n";

        // collect values
        for (auto const &iti : vecITI) {
            dataSources.push_back(iti.dataSourcesLE);
            seeds.push_back(iti.seed);
            ids_graph_traversals.push_back(id_GT);
        }
        id_GT++;
    }
    file << "// ===================================\n\n";

    // Determine number of values
    // Fill a vector with indices 0, 1, ..., nbValues-1
    size_t nbValues = dataSources.size();
    std::vector<size_t> indices(nbValues);
    std::iota(indices.begin(), indices.end(), 0);

    // Randomly shuffle indices if requested
    if (randomize) {
        unsigned int seed = 0; 
        std::mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);
    }

    // Write Header
    file << "#ifndef SEEDS_H\n"
    << "#define SEEDS_H\n\n"
    << "#define NB_SEED " << nbValues << "\n"
    << "#define NB_VALUES_PER_CLASS " << NB_VALUES_PER_CLASS << "\n\n";


    // Write dataSourcesLE arrays (split into per-feature arrays)
    for (size_t featureIdx = 0; featureIdx < nbDataSources; ++featureIdx) 
    {
        file << "static const double dataSourcesLE_" << featureIdx << "[NB_SEED] = {";
        
        for (size_t i = 0; i < indices.size(); i++)
        {
            if (i > 0){
                file << ", ";
            }
            
            file << dataSources[indices[i]][featureIdx];
        }
            
        file << "};\n";
    }


    // Write seeds
    file << "static const uint64_t seeds[NB_SEED] = {";
    for (size_t i = 0; i < indices.size(); i++)
    {
        if (i > 0){
            file << ", ";
        }
        
        file << seeds[indices[i]];
        
    }
    
    file << "};\n";


    // Write ids_graph_traversals
    
    file << "static const unsigned int ids_graph_traversals[NB_SEED] = {";
    
    for (size_t i = 0; i < indices.size(); i++)
    {
        if (i > 0)
        {
            file << ", ";
        }
        file << ids_graph_traversals[indices[i]];
    }
    file << "};\n";


    file << "\n#endif // SEEDS_H\n";

    file.close();

    if (randomize)
    {
        std::cout << "Data order was randomized." << std::endl;
    }
    std::cout << "Data written to " << filename << " successfully." << std::endl;
}
