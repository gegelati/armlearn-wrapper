
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

            // Update limits
            auto iter = trainingValidationResults.begin();
            std::advance(iter, trainingValidationResults.size() - 1);
            double bestResult = iter->first->getResult();
            ((ArmLearnWrapper&)learningEnvironment).updateCurrentLimits(bestResult, params.nbIterationsPerPolicyEvaluation);
        }

    }
    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }

    
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