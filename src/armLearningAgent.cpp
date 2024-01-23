
#include "armLearningAgent.h"
#include "armLearnLogger.h"
#include "ArmLearnWrapper.h"

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
    double bestResult = iter->first->getResult();

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
    // TODO Changes for limits update without training validation
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
        bestResult = iter->first->getResult();

    }

    if (doUpdateLimits){
        // Log limits
        if(typeid(learningEnvironment) == typeid(ArmLearnWrapper)){
            for (auto logger : loggers) {
                if(typeid(logger.get()) == typeid(Log::ArmLearnLogger)){
                    ((Log::ArmLearnLogger&)logger.get()).logEnvironnementStatus(
                        ((ArmLearnWrapper&)learningEnvironment).getCurrentMaxLimitTarget(),
                        ((ArmLearnWrapper&)learningEnvironment).getCurrentMaxLimitStartingPos()
                    );
                }
            }

            ((ArmLearnWrapper&)learningEnvironment).updateCurrentLimits(bestResult, params.nbIterationsPerPolicyEvaluation);
        }
    }
    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }

    
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

    // Init results
    double result = 0.0;

    // Evaluate nbIteration times
    for (auto iterationNumber = 0; iterationNumber < this->params.nbIterationsPerPolicyEvaluation; iterationNumber++) {
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

        // Update results
        result += le.getScore();
    }

    // Create the EvaluationResult
    auto evaluationResult =
        std::shared_ptr<EvaluationResult>(new EvaluationResult(
            result / (double)params.nbIterationsPerPolicyEvaluation,
            params.nbIterationsPerPolicyEvaluation));

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