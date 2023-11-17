
#ifndef ARM_LEARN_LOGGER_H
#define ARM_LEARN_LOGGER_H

#include <iomanip>

#include "gegelati.h"

namespace Log {

    /**
     * \brief Basic logger that will display some useful information
     *
     * The information logged by this LALogger are generation number, nb of
     * vertices, min, mean, avg score of this generation and to finish some
     * timing. Everything is logged like a tab with regularly spaced columns.
     */
    class ArmLearnLogger : public LALogger
    {
      private:
        /**
         * Width of columns when logging values.
         */
        int colWidth = 12;

        /**
         * \brief Logs the min, avg and max score of the generation.
         *
         * This method is used by the eval and valid callback as
         * they both have the same input and want to log the same elements
         * (min, avg max).
         */
        void logResults(std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                                      const TPG::TPGVertex*>& results);

      public:
        /**
         * \brief Same constructor as LaLogger. Default output is cout.
         *
         * \param[in] la LearningAgent whose information will be logged by the
         * LABasicLogger.
         * \param[in] out The output stream the logger will send
         * elements to.
         */
        explicit ArmLearnLogger(Learn::LearningAgent& la,
                               std::ostream& out = std::cout)
            : LALogger(la, out)
        {
            // fixing float precision
            *this << std::setprecision(2) << std::fixed << std::right;
            this->logHeader();
        }

        /**
         * Inherited via LaLogger
         *
         * \brief Logs the header (column names) of the tab that will be logged.
         */
        virtual void logHeader() override;

        /**
         * Inherited via LALogger.
         *
         * \brief Logs the generation of training.
         *
         * \param[in] generationNumber The number of the current
         * generation.
         */
        virtual void logNewGeneration(uint64_t& generationNumber) override;

        /**
         * Inherited via LALogger.
         *
         * \brief Logs the vertices nb of the tpg.
         */
        virtual void logAfterPopulateTPG() override;

        /**
         * Inherited via LaLogger.
         *
         * \brief Logs the min, avg and max score of the generation.
         *
         * If doValidation is true, it only updates eval time.
         * The method logResults will be called in order to log
         * statistics about results (method shared with logAfterValidate).
         *
         * \param[in] results scores of the evaluation.
         */
        virtual void logAfterEvaluate(
            std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                          const TPG::TPGVertex*>& results) override;

        /**
         * Inherited via LaLogger.
         *
         * \brief Does nothing in this logger.
         */
        virtual void logAfterDecimate() override{
            // nothing to log
        };

        /**
         * Inherited via LaLogger.
         *
         * \brief Logs the min, avg and max score of the generation.
         *
         * If doValidation is true, no eval results are logged so that
         * the logger can only show validation results.
         *
         * \param[in] results scores of the validation.
         */
        virtual void logAfterValidate(
            std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                          const TPG::TPGVertex*>& results) override;

        /**
         * Inherited via LaLogger
         *
         * \brief Logs the eval, valid (if doValidation is true)
         * and total running time.
         */
        virtual void logEndOfTraining() override;

        /**
         * \brief Logs the size of the environnement
         * 
         * \param envSizeTargets Size of the environnement for the targets
         * \param envSizeStartingPos Size of the environnement for the starting positions
         */
        void logEnvironnementStatus(double envSizeTargets, double envSizeStartingPos);

    };

} // namespace Log

#endif
