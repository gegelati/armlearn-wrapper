
#ifndef ARM_LEARNING_AGENT
#define ARM_LEARNING_AGENT

#include <mutex>
#include <queue>
#include <thread>

#include <gegelati.h>
#include "trainingParameters.h"

namespace Learn {
    /**
     * \brief  Class used to control the learning steps of a TPGGraph within
     * a given LearningEnvironment, with parallel executions for speedup
     * purposes.
     *
     * This class is intented to replace the default LearningAgent soon.
     *
     * Because of parallelism, determinism of the LearningProcess could easiliy
     * be lost, but this implementation must remain deterministic at all costs.
     */
    class ArmLearningAgent : public ParallelLearningAgent
    {
      private:
        std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> bestTrainingResult;

        /// Parameters for the trianing
        TrainingParameters& trainingParams;

        bool doUpdateLimits;
        bool doTrainingValidation;

        /// Vector that contain the best results of the five last generation
        std::vector<double> fiveLastBest;
        
      public:
        /**
         * \brief Constructor for ParallelLearningAgent.
         *
         * Based on default constructor of LearningAgent
         *
         * \param[in] le The LearningEnvironment for the TPG.
         * \param[in] iSet Set of Instruction used to compose Programs in the
         *            learning process.
         * \param[in] p The LearningParameters for the LearningAgent.
         * \param[in] factory The TPGFactory used to create the TPGGraph. A
         * default TPGFactory is used if none is provided.
         */
        ArmLearningAgent(
            LearningEnvironment& le, const Instructions::Set& iSet,
            const LearningParameters& p, TrainingParameters& trainingParams,
            const TPG::TPGFactory& factory = TPG::TPGFactory())
            : ParallelLearningAgent(le, iSet, p, factory), trainingParams(trainingParams) {
              this->doUpdateLimits = (this->trainingParams.progressiveModeTargets || this->trainingParams.progressiveModeStartingPos);
              this->doTrainingValidation = (this->trainingParams.doTrainingValidation && this->doUpdateLimits);
            };

        /**
         * \brief Evaluate all root TPGVertex of the TPGGraph.
         *
         * **Replaces the function from the base class LearningAgent.**
         *
         * This method must always the same results as the evaluateAllRoots for
         * a sequential execution. The Archive should also be updated in the
         * exact same manner.
         *
         * This method calls the evaluateJob method for every root TPGVertex
         * of the TPGGraph. The method returns a sorted map associating each
         * root vertex to its average score, in ascending order or score.
         *
         * \param[in] generationNumber the integer number of the current
         * generation. \param[in] mode the LearningMode to use during the policy
         * evaluation.
         */
        virtual void trainOneGeneration(uint64_t generationNumber) override;

        void testingBestRoot(uint64_t generationNumber);

        /**
         * \brief Evaluates policy starting from the given root.
         *
         * The policy, that is, the TPGGraph execution starting from the given
         * TPGVertex is evaluated nbIteration times. The generationNumber is
         * combined with the current iteration number to generate a set of
         * seeds for evaluating the policy.
         *
         * The method is const to enable potential parallel calls to it.
         *
         * \param[in] tee The TPGExecutionEngine to use.
         * \param[in] job The job containing the root and archiveSeed for
         * the evaluation.
         * \param[in] generationNumber the integer number of the current
         * generation.
         * \param[in] mode the LearningMode to use during the policy
         * evaluation.
         * \param[in] le Reference to the LearningEnvironment to use
         * during the policy evaluation (may be different from the attribute of
         * the class in child LearningAgentClass).
         *
         * \return a std::shared_ptr to the EvaluationResult for the root. If
         * this root was already evaluated more times then the limit in
         * params.maxNbEvaluationPerPolicy, then the EvaluationResult from the
         * resultsPerRoot map is returned, else the EvaluationResult of the
         * current generation is returned, already combined with the
         * resultsPerRoot for this root (if any).
         */
        virtual std::shared_ptr<EvaluationResult> evaluateJob(
            TPG::TPGExecutionEngine& tee, const Job& job,
            uint64_t generationNumber, LearningMode mode,
            LearningEnvironment& le) const override;


        virtual std::queue<std::shared_ptr<Learn::Job>> makeJobs(
        Learn::LearningMode mode, TPG::TPGGraph* tpgGraph) override;
    };
} // namespace Learn
#endif
