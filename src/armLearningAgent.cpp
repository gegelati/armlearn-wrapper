
#include "armLearningAgent.h"
#include "armLearnLogger.h"
#include "ArmLearnWrapper.h"
#include "armlearnEvaluationResult.h"

void Learn::ArmLearningAgent::trainOneGeneration(uint64_t generationNumber){

    for (auto logger : loggers) {
        logger.get().logNewGeneration(generationNumber);
    }

    
    // Populate Sequentially
    Mutator::TPGMutator::populateTPG(*this->tpg, this->archive,
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
    double bestResult = std::dynamic_pointer_cast<Learn::ArmlearnEvaluationResult>(iter->first)->getDistance();

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
    // Update the best
    this->updateEvaluationRecords(results);

    for (auto logger : loggers) {
        logger.get().logAfterDecimate();
    }

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
    }

    // Training Validation
    if (doTrainingValidation){

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

    if (doUpdateLimits){
        // Log limits
        if(typeid(learningEnvironment) == typeid(ArmLearnWrapper)){
            double target = (trainingParams.progressiveRangeTarget) ? ((ArmLearnWrapper&)learningEnvironment).getCurrentRangeTarget() : ((ArmLearnWrapper&)learningEnvironment).getCurrentMaxLimitTarget();
            for (auto logger : loggers) {
                if(typeid(logger.get()) == typeid(Log::ArmLearnLogger)){
                    ((Log::ArmLearnLogger&)logger.get()).logEnvironnementStatus(target, ((ArmLearnWrapper&)learningEnvironment).getCurrentMaxLimitStartingPos());
                }
            }

            // Update the current limits
            ((ArmLearnWrapper&)learningEnvironment).updateCurrentLimits(bestResult, params.nbIterationsPerPolicyEvaluation);
        }
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

    // Evaluate nbIteration times
    for (auto iterationNumber = 0; iterationNumber < nbIteration && !cancelThisRoot; iterationNumber++) {

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
            // Get the action
            uint64_t actionID =
                ((const TPG::TPGAction*)tee.executeFromRoot(*root).back())
                    ->getActionID();
            // Do it
            le.doAction(actionID);
            // Count actions
            nbActions++;

        }

        // Update score
        score += le.getScore();

        distance += ((ArmLearnWrapper&)le).getDistance();

        if(((ArmLearnWrapper&)le).getDistance() < trainingParams.rangeTarget){
            success += 1;
        }

        // If we past the generation 5 and iteration number is above 10
        if (generationNumber > 5 && iterationNumber > trainingParams.nbIterationRootCanceled && mode == LearningMode::TRAINING){

            // If the current mean distance times 3 is above the mean current best, then the root is canceled
            if (distance/(iterationNumber+1) > 2*std::accumulate(fiveLastBest.begin(), fiveLastBest.end(), 0.0)/fiveLastBest.size()){
                cancelThisRoot = true;
                nbIteration = iterationNumber + 1;
            }
        }


        // Push back the id with the score/distance
        trajectoriesScore.push_back(std::make_pair(iterationNumber, (trainingParams.isScoreResult) ? -1 * ((ArmLearnWrapper&)le).getDistance(): le.getScore()));
    }

    if(trainingParams.testing){
        ((ArmLearnWrapper&)le).logTestingTrajectories(true);
    }

    // Create the EvaluationResult
    auto evaluationResult =
        std::shared_ptr<Learn::ArmlearnEvaluationResult>(new Learn::ArmlearnEvaluationResult(
            score / (double)nbIteration,
            success / (double)nbIteration,
            distance / (double)nbIteration,
            trajectoriesScore, nbIteration,
            trainingParams.isScoreResult));

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