
#ifndef ARMLEARN_EVALUATION_RESULT_H
#define ARMLEARN_EVALUATION_RESULT_H

#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "learn/evaluationResult.h"

namespace Learn {
    /**
     * \brief Class for storing all results of a policy evaluation in
     * in adversarial mode with an AdversarialLearningEnvironment.
     *
     * The main difference with the base EvaluationResult class is that there
     * are several results. Indeed, in adversarial mode there are several roots
     * in a single simulation and, as a consequence, there are several results
     * at the end.
     */
    class ArmlearnEvaluationResult : public EvaluationResult
    {
      protected:
        /// The scores of the roots, in the order in which they participated.
        double distance = 0;
        double success = 0;
        double nbActivatedTeam = 0;
        double nbActivatedAction = 0;
        double nbActivatedTeamRatio = 0;
        double nbCollision = 0;

        std::vector<std::pair<int, double>> trajScores;

      public:
        /**
         * \brief Base constructor of EvaluationResult, allowing to set scores
         * and the number of evaluations.
         *
         * @param[in] res The scores of the roots in the order.
         * @param[in] nbEval The number of evaluations that have been done to
         * get these scores. Default is 1 as we can guess user only did 1
         * iteration.
         */
        ArmlearnEvaluationResult(const double score, const double success, const double distance, const double nbActivatedTeam, 
        const double nbActivatedAction, const double nbActivatedTeamRatio, double nbCollision, std::vector<std::pair<int, double>> trajScores, const size_t& nbEval)
            : EvaluationResult(score, nbEval)
        {
          this->success = success;
          this->distance = distance;
          this->nbActivatedTeam = nbActivatedTeam;
          this->nbActivatedAction = nbActivatedAction;
          this->nbActivatedTeamRatio = nbActivatedTeamRatio;
          this->trajScores = trajScores;
        }

        double getSuccess() const;

        double getDistance() const;

        double getNbActivatedTeam() const;

        double getNbActivatedAction() const;

        double getNbActivatedTeamRatio() const;

        double getNbCollision() const;

        std::vector<std::pair<int, double>> getTrajScores();

        virtual EvaluationResult& operator+=(const EvaluationResult& other) override;

    };
} // namespace Learn

#endif
