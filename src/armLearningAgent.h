
#ifndef ARM_LEARNING_AGENT
#define ARM_LEARNING_AGENT

#include <mutex>
#include <queue>
#include <thread>

#include <gegelati.h>

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

        bool doTrainingValidation;
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
            const LearningParameters& p, bool doTrainingValidation,
            const TPG::TPGFactory& factory = TPG::TPGFactory())
            : ParallelLearningAgent(le, iSet, p, factory) {
              this->doTrainingValidation = doTrainingValidation;
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

        virtual std::queue<std::shared_ptr<Learn::Job>> makeJobs(
        Learn::LearningMode mode, TPG::TPGGraph* tpgGraph) override;
    };
} // namespace Learn
#endif
