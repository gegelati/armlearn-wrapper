
#include <iomanip>
#include <numeric>

#include "armLearnLogger.h"

void Log::ArmLearnLogger::logResults(
    std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                  const TPG::TPGVertex*>& results)
{
    auto iter = results.begin();
    double min = iter->first->getResult();
    std::advance(iter, results.size() - 1);
    double max = iter->first->getResult();
    
    double avg = std::accumulate(
        results.begin(), results.end(), 0.0,
        [](double acc,
           std::pair<std::shared_ptr<Learn::EvaluationResult>,
                     const TPG::TPGVertex*>
               pair) -> double { return acc + pair.first->getResult(); });
    avg /= (double)results.size();
    *this << std::setw(colWidth) << min << std::setw(colWidth) << avg
          << std::setw(colWidth) << max;
}

void Log::ArmLearnLogger::logHeader()
{
    // First line of header
    //*this << std::left;
    *this << std::setw(2 * colWidth) << " " << std::setw(colWidth) << "Train";
    if (doValidation) {
        *this << std::setw(2 * colWidth) << " " << std::setw(1 * colWidth)
              << "Valid";
    }
    if (doTrainingValidation) {
        *this << std::setw(2 * colWidth) << " " << std::setw(1 * colWidth)
              << "Train Valid";
    }
    *this << std::endl;

    // Second line of header
    //*this << std::right;
    *this << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "NbVert"
          << std::setw(colWidth) << "Tmin" << std::setw(colWidth) << "Tavg"
          << std::setw(colWidth) << "Tmax";
    if (doValidation) {
        *this << std::setw(colWidth) << "Vmin" << std::setw(colWidth) << "Vavg"
              << std::setw(colWidth) << "Vmax";
    }

    if (doTrainingValidation) {
        *this << std::setw(colWidth) << "TVmin" << std::setw(colWidth) << "TVavg"
              << std::setw(colWidth) << "TVmax";
    }

    if (doUpdateLimits){
        *this << std::setw(colWidth) << "S_Targ"; 
        *this << std::setw(colWidth) << "S_StartP";
    }
        

    *this << std::setw(colWidth) << "T_mutat" << std::setw(colWidth)
          << "T_eval";
    if (doValidation) {
        *this << std::setw(colWidth) << "T_val";
    }
    if (doTrainingValidation) {
        *this << std::setw(colWidth) << "T_TrVal";
    }
    *this << std::setw(colWidth) << "T_total"; 


    *this << std::endl;
    
}

void Log::ArmLearnLogger::logNewGeneration(uint64_t& generationNumber)
{
    *this << std::setw(colWidth) << generationNumber;
    // resets checkpoint to be able to show evaluation time
    chronoFromNow();
}

void Log::ArmLearnLogger::logAfterPopulateTPG()
{
    this->mutationTime = getDurationFrom(*checkpoint);

    *this << std::setw(colWidth)
          << this->learningAgent.getTPGGraph()->getNbVertices();

    chronoFromNow();
    
}

void Log::ArmLearnLogger::logAfterEvaluate(
    std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                  const TPG::TPGVertex*>& results)
{
    evalTime = getDurationFrom(*checkpoint);

    logResults(results);
    
    // resets checkpoint to be able to show validation time if there is some
    chronoFromNow();
}

void Log::ArmLearnLogger::logAfterValidate(
    std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                  const TPG::TPGVertex*>& results)
{
    validTime = getDurationFrom(*checkpoint);

    // being in this method means validation is active, and so we are sure we
    // can log results
    logResults(results);

    chronoFromNow();
}

void Log::ArmLearnLogger::logAfterTrainingValidate(
    std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                  const TPG::TPGVertex*>& results)
{
    trainingValidTime = getDurationFrom(*checkpoint);

    // being in this method means training validation is active, and so we are sure we
    // can log results
    logResults(results);
}

void Log::ArmLearnLogger::logEndOfTraining()
{
    *this << std::setw(colWidth) << mutationTime;
    *this << std::setw(colWidth) << evalTime;
    if (doValidation) {
        *this << std::setw(colWidth) << validTime;
    }
    if (doTrainingValidation){
        *this << std::setw(colWidth) << trainingValidTime;
    }
    *this << std::setw(colWidth) << getDurationFrom(*start) << std::endl;
}

void Log::ArmLearnLogger::logEnvironnementStatus(double envSizeTargets, double envSizeStartingPos)
{
    *this << std::setw(colWidth) << envSizeTargets;
    *this << std::setw(colWidth) << envSizeStartingPos;
}
