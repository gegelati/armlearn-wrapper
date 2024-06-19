
#include "armLearningAgent.h"
#include "armLearnLogger.h"
#include "ArmLearnWrapper.h"
#include "armlearnEvaluationResult.h"

void Learn::ArmLearningAgent::trainOneGeneration(uint64_t generationNumber){

    for (auto logger : loggers) {
        logger.get().logNewGeneration(generationNumber);
    }

    // Populate Sequentially
    MARL::MarlTPGMutator::populateTPG(*dynamic_cast<MARL::MarlTPGGraph*>(this->tpg.get()), this->archive,
                                     this->params.mutation, this->rng,
                                     maxNbThreads);
    for (auto logger : loggers) {
        logger.get().logAfterPopulateTPG();
    }


    // Evaluate
    auto results =
        this->evaluateAllRoots(generationNumber, LearningMode::TRAINING);
    for (auto logger : loggers) {
        logger.get().logAfterEvaluate(results);
    }
    auto iter = results.begin();
    std::advance(iter, results.size() - 1);
    double bestResult = std::dynamic_pointer_cast<Learn::ArmlearnEvaluationResult>(iter->first)->getResult();
    // Update five last best score
    fiveLastBest.push_back(bestResult);
    if (generationNumber >= 5){
        fiveLastBest.erase(fiveLastBest.begin());
    }
    for(auto pair: std::dynamic_pointer_cast<Learn::ArmlearnEvaluationResult>(iter->first)->getTrajScores()){
        ((ArmLearnWrapper&)learningEnvironment).addToScoreTrajectories(pair.first, pair.second);
    }
    // Remove worst performing roots
    decimateWorstRoots(results);

    // Clear best results
    bestTrainingResult.clear();
    // Utilisez un itérateur pour parcourir la std::multimap d'origine à partir de la fin
    auto it = results.rbegin();
    for (int i = 0; i < 5 && it != results.rend(); ++i, ++it) {
        // Ajoutez les éléments à la nouvelle std::multimap
        bestTrainingResult.insert(*it);
    }
    // Does a validation or not according to the parameter doValidation
    if (params.doValidation) {
        auto validationResults  = this->evaluateAllRoots(generationNumber, LearningMode::VALIDATION);
        
        for (auto logger : loggers) {
            logger.get().logAfterValidate(validationResults);
        }
        // Update the best
        this->updateEvaluationRecords(validationResults);
    } else{
        
        // Update the best
        this->updateEvaluationRecords(results);
    }

    // Training Validation
    if (trainingParams.doTrainingValidation){

        auto trainingValidationResults = this->evaluateAllRoots(generationNumber, LearningMode::TESTING);
        for (auto logger : loggers) {
            if(typeid(logger.get()) == typeid(Log::ArmLearnLogger)){
                ((Log::ArmLearnLogger&)logger.get()).logAfterTrainingValidate(trainingValidationResults);
            }
        }

        // Update limits
        auto iter = trainingValidationResults.begin();
        std::advance(iter, trainingValidationResults.size() - 1);
        bestResult = std::dynamic_pointer_cast<Learn::ArmlearnEvaluationResult>(iter->first)->getDistance();

    }


    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }

    
}

void Learn::ArmLearningAgent::testingBestRoot(uint64_t generationNumber){

    std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;
    checkpoint = std::make_shared<std::chrono::time_point<
    std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());

    auto mode = Learn::LearningMode::VALIDATION;

    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
            this->env, NULL);

    auto roots = tpg->getRootVertices();

    auto job = makeJob(roots.at(0), mode);
    this->archive.setRandomSeed(job->getArchiveSeed());
    std::shared_ptr<EvaluationResult> result = this->evaluateJob(
        *tee, *job, generationNumber, mode, this->learningEnvironment);



    std::cout<<"Testing score : "<<result->getResult();
    std::cout << " -- Testing success rate " << std::dynamic_pointer_cast<ArmlearnEvaluationResult>(result)->getSuccess();

    auto testingTime = ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();
    std::cout<<" -- Time of testing "<<testingTime<<std::endl;



}

std::shared_ptr<Learn::EvaluationResult> Learn::ArmLearningAgent::evaluateJob(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le) const
{
 
 
    // Get the tpg execution engine with the right class
    if(!dynamic_cast<MARL::MarlTpgExecutionEngine*>(&tee)){
        throw std::runtime_error("tee should be a MarlTpgExecutionEngine object but "
                                 "seems to be a simple TPGExecutionEngine object");
    }
    MARL::MarlTpgExecutionEngine* marlTee = dynamic_cast<MARL::MarlTpgExecutionEngine*>(&tee);

    // Get the learning environment with the right class
    if(!dynamic_cast<MARL::MarlLearningEnvironment*>(&le)){
        throw std::runtime_error("le should be a MarlLearningEnvironment object but "
                                 "seems to be a simple LearningEnvironment object");
    }
    MARL::MarlLearningEnvironment* marlLe = dynamic_cast<MARL::MarlLearningEnvironment*>(&le);
 
    // Only consider the first root of jobs as we are not in adversarial mode
    const TPG::TPGVertex* root = job.getRoot();

    // Skip the root evaluation process if enough evaluations were already
    // performed. In the evaluation mode only.
    std::shared_ptr<Learn::EvaluationResult> previousEval;
    if (mode == LearningMode::TRAINING &&
        this->isRootEvalSkipped(*root, previousEval)) {
        return previousEval;
    }

    bool cancelThisRoot = false;

    double success = 0.0;

    double distance = 0.0;

    std::vector<std::pair<int, double>> trajectoriesScore;

    // Init Score
    double score = 0.0;

    uint64_t nbIteration = (mode == LearningMode::TRAINING) ? trainingParams.nbIterationTraining : this->params.nbIterationsPerPolicyEvaluation;

    
    double nbActivatedTeam = 0.0;
    double nbActivatedAction = 0.0;
    double nbActivatedTeamRatio = 0.0;

    // Evaluate nbIteration times
    for (auto iterationNumber = 0; iterationNumber < nbIteration && !cancelThisRoot; iterationNumber++) {


        double nbActivatedTeamOneEp = 0.0;
        double nbActivatedActionOneEp = 0.0;
        double nbActivatedTeamRatioOneEp = 0.0;


        if(trainingParams.testing){
            std::cout<<"Episode "<<iterationNumber+1<<"/"<<nbIteration<<"      "<<std::flush;
            std::cout << '\r';
        }


        // Compute a Hash
        Data::Hash<uint64_t> hasher;
        uint64_t hash = hasher(generationNumber) ^ hasher(iterationNumber);

        // Reset the learning Environment
        le.reset(hash, mode, iterationNumber, generationNumber);

        uint64_t nbActions = 0;
        while (!le.isTerminal() &&
               nbActions < this->params.maxNbActionsPerEval) {
            


            std::vector<std::uint64_t> actionsID;
            if((dynamic_cast<const MARL::MarlTPGTeam*>(root))){
                // Get the actions
                std::map<std::uint64_t, std::pair<std::uint64_t, double>> actions 
                    = marlTee->executeFromRoot(*root, marlLe->getInitActions(), this->params);


                nbActivatedTeamRatioOneEp += (dynamic_cast<const MARL::MarlTPGTeam*>(root))->getNbActivateTeamRatio();
                nbActivatedTeamOneEp += (dynamic_cast<const MARL::MarlTPGTeam*>(root))->getNbActivateTeam();
                nbActivatedActionOneEp += (dynamic_cast<const MARL::MarlTPGTeam*>(root))->getNbActivateAction();

                // Browse the map to get the actions ID
                for (const auto& obj :actions) {
                    actionsID.push_back(obj.second.first);
                }
            }else if((dynamic_cast<const MARL::MarlTPGAction*>(root))){
                actionsID = marlLe->getInitActions();
                const MARL::MarlTPGAction* actionRoot = (dynamic_cast<const MARL::MarlTPGAction*>(root));
                actionsID[actionRoot->getActionID()] = actionRoot->getActionValue();
            }else {
                throw std::runtime_error("Root should be either MARL Team or MARL Action");
            }



            
            // Do it
            marlLe->doActions(actionsID);
            // Count actions
            nbActions++;

        }

        nbActivatedTeamRatio += nbActivatedTeamRatioOneEp / nbActions;
        nbActivatedTeam += nbActivatedTeamOneEp / nbActions;
        nbActivatedAction += nbActivatedActionOneEp / nbActions;

        // Update score
        score += le.getScore();
        //std::cout<<"Score : "<<le.getScore()<<std::endl;

        distance += ((ArmLearnWrapper&)le).getDistance();

        if(((ArmLearnWrapper&)le).getDistance() < trainingParams.rangeTarget){
            success += 1;
        }

        // Push back the id with the score
        trajectoriesScore.push_back(std::make_pair(iterationNumber, le.getScore()));
    }

    double meanScore = score / (double)nbIteration;
    if(trainingParams.meanScoreWithStd){
        double sumVar = 0.0;
        for (const auto& pair : trajectoriesScore) {
            double diff = pair.second - meanScore;
            sumVar += diff * diff;
        }
        double std = std::sqrt(sumVar / (double)nbIteration);
        meanScore -= std;
    }

    //std::cout<<"Mean Score : "<<meanScore<<std::endl;

    if(trainingParams.testing){
        ((ArmLearnWrapper&)le).logTestingTrajectories(true);
    }

    // Create the EvaluationResult
    auto evaluationResult =
        std::shared_ptr<Learn::ArmlearnEvaluationResult>(new Learn::ArmlearnEvaluationResult(
            meanScore,
            success / (double)nbIteration,
            distance / (double)nbIteration,
            nbActivatedTeam / (double)nbIteration,
            nbActivatedAction / (double)nbIteration,
            nbActivatedTeamRatio / (double)nbIteration,
            trajectoriesScore, nbIteration));

    // Combine it with previous one if any
    if (previousEval != nullptr) {
        *evaluationResult += *previousEval;
    }
    return evaluationResult;
}

std::queue<std::shared_ptr<Learn::Job>> Learn::ArmLearningAgent::makeJobs(
    Learn::LearningMode mode, TPG::TPGGraph* tpgGraph)
{
    // sets the tpg to the Learning Agent's one if no one was specified
    tpgGraph = tpgGraph == nullptr ? tpg.get() : tpgGraph;

    std::queue<std::shared_ptr<Learn::Job>> jobs;
    if(mode == Learn::LearningMode::TRAINING){
        auto roots = tpgGraph->getRootVertices();
        for (int i = 0; i < roots.size(); i++) {
            auto job = makeJob(roots.at(i), mode, i);
            jobs.push(job);
        }
    }else{
        int index = 0;
        for (auto pairRoot: bestTrainingResult) {
            auto job = makeJob(pairRoot.second, mode, index);
            jobs.push(job);
            index++;
        }
    }

    return jobs;
}

std::vector<const TPG::TPGVertex *> Learn::ArmLearningAgent::keepBestPolicies(uint64_t nbPolicies)
{
    // Some actions may be encountered but not removed while scanning the
    // results map they should be re-inserted to the list before leaving the
    // method.
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>
        preservedActionRoots;

    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> results;

    while(this->tpg->getNbRootVertices() != nbPolicies){

        auto currentNbRoots = this->tpg->getNbRootVertices();
        auto i = 0;
        results = this->evaluateAllRoots(params.nbGenerations+1, LearningMode::TRAINING);

        while (i < currentNbRoots - nbPolicies && results.size() > 0) {
            // If the root is an action, do not remove it!
            const TPG::TPGVertex* root = results.begin()->second;

            tpg->removeVertex(*results.begin()->second);
            // Removed stored result (if any)
            this->resultsPerRoot.erase(results.begin()->second);

            results.erase(results.begin());

            // Increment loop counter
            i++;
        }
        // Restore root actions
        results.insert(preservedActionRoots.begin(), preservedActionRoots.end());
    } 

    // Conserve only the roots, the results do not are usefull now
    std::vector<const TPG::TPGVertex *> bestRoots;
    for(auto pair: results){
        bestRoots.push_back(pair.second);   
    }
    return bestRoots;
}


std::multimap<const TPG::TPGVertex *, std::multimap<double, bool>> Learn::ArmLearningAgent::generateDataOfRoots(std::vector<const TPG::TPGVertex *>& bestRoots, LearningEnvironment& le, uint64_t nbIterations){


    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
        this->env, NULL);

    // Get the tpg execution engine with the right class
    if (!dynamic_cast<MARL::MarlTpgExecutionEngine*>(tee.get())) {
        throw std::runtime_error("tee should be a MarlTpgExecutionEngine object but "
                                "seems to be a simple TPGExecutionEngine object");
    }
    MARL::MarlTpgExecutionEngine* marlTee = dynamic_cast<MARL::MarlTpgExecutionEngine*>(tee.get());

    // Get the learning environment with the right class
    if(!dynamic_cast<MARL::MarlLearningEnvironment*>(&le)){
        throw std::runtime_error("le should be a MarlLearningEnvironment object but "
                                 "seems to be a simple LearningEnvironment object");
    }
    MARL::MarlLearningEnvironment* marlLe = dynamic_cast<MARL::MarlLearningEnvironment*>(&le);


    // Instantiate the data map
    std::multimap<const TPG::TPGVertex *, std::multimap<double, bool>> data;

    for(const TPG::TPGVertex * root: bestRoots){

        // To store the score and success
        std::multimap<double, bool> dataRoot;

        // Evaluate nbIteration times
        for (auto iterationNumber = 0; iterationNumber < nbIterations; iterationNumber++) {

            // Compute a Hash
            Data::Hash<uint64_t> hasher;
            uint64_t hash = hasher(params.nbGenerations+1) ^ hasher(iterationNumber); //TODO

            // Reset the learning Environment
            le.reset(hash, Learn::LearningMode::TESTING, iterationNumber, 1000);

            uint64_t nbActions = 0;
            while (!le.isTerminal() &&
                nbActions < this->params.maxNbActionsPerEval) {
                


                std::vector<std::uint64_t> actionsID;
                if((dynamic_cast<const MARL::MarlTPGTeam*>(root))){
                    // Get the actions
                    std::map<std::uint64_t, std::pair<std::uint64_t, double>> actions 
                        = marlTee->executeFromRoot(*root, marlLe->getInitActions(), this->params);

                    // Browse the map to get the actions ID
                    for (const auto& obj :actions) {
                        actionsID.push_back(obj.second.first);
                    }
                }else if((dynamic_cast<const MARL::MarlTPGAction*>(root))){
                    actionsID = marlLe->getInitActions();
                    const MARL::MarlTPGAction* actionRoot = (dynamic_cast<const MARL::MarlTPGAction*>(root));
                    actionsID[actionRoot->getActionID()] = actionRoot->getActionValue();
                }else {
                    throw std::runtime_error("Root should be either MARL Team or MARL Action");
                }



                
                // Do it
                marlLe->doActions(actionsID);
                // Count actions
                nbActions++;

            }

            // Save score
            double score = le.getScore();

            // Save success
            bool success = ((ArmLearnWrapper&)le).getDistance() < trainingParams.rangeTarget;

            // Push back the id with the score
            dataRoot.insert(std::make_pair(score, success));
        }
        data.insert(std::make_pair(root, dataRoot));

    }

    return data;


}

void Learn::ArmLearningAgent::createPopulationFromRoots(std::vector<const TPG::TPGVertex *> roots){

    // Delete all roots except actions
    std::vector<const TPG::TPGVertex*> rootToDelete;

    for(const TPG::TPGVertex* root : this->tpg->getRootVertices()){
        if(dynamic_cast<const MARL::MarlTPGAction*>(root)== nullptr){
            rootToDelete.push_back(root);
        }
    }
    for(const TPG::TPGVertex* root: rootToDelete){
        this->tpg->removeVertex(*root);
    }

    std::multimap<const TPG::TPGVertex*, const TPG::TPGVertex*> addedTeams;
    for(auto root: roots){ 
        dynamic_cast<MARL::MarlTPGGraph*>(this->tpg.get())->graftRoot(*root, addedTeams);
    }
}