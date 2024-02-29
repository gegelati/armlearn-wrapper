#ifndef ARM_LEARN_WRAPPER_H
#define ARM_LEARN_WRAPPER_H

#include <random>

#include <gegelati.h>
#include <armlearn/nowaitarmsimulator.h>
#include <armlearn/serialcontroller.h>
#include <armlearn/trajectory.h>

#include <armlearn/widowxbuilder.h>
#include <armlearn/basiccartesianconverter.h>
#include <armlearn/devicelearner.h>
#include "trainingParameters.h"

/**
* LearningEnvironment to use armLean in order to learn how to move a robotic arm.
* It inherits from LearningEnvironment to be used by Gegelati as an environment,
* and from DeviceLearner to access motors of the arm
* This double inheritance results in useless methods but it isn't a real problem.
*
* At the beginning, the arm is placed at startingPos. The goal will be to reach a target :
* targets[0]. targets is a list of targets that can rotate at each new simulation
* (see method reset)
* At each frame, the TPG is asked to take a decision based on the input
* (currently pos of motors and distances between arm and target on x, y and z).
* It decides to move a motor in one direction from a fix
* angle, or to do nothing. Then, its inputs are updated and it goes on.
* The simulation is stopped after something like 1000 frames, and we
* take the negative distance arm-target as score.
*/
class ArmLearnWrapper : public Learn::LearningEnvironment, armlearn::learning::DeviceLearner {
protected:

    bool printing=false;
    void computeInput();

    double computeReward(bool givePenaltyMoveUnavailable);

    TrainingParameters& params;

    /// true if the the environnement is terminated
    bool terminal = false;

    /// @brief Boolean used to control whether the LE includes the 2 servos
    /// controlling the rotation of the hand and the fingers.
    bool handServosTrained;

    /// Randomness control
    Mutator::RNG rng;

    /// Current motor position 
    Data::PrimitiveTypeArray<double> motorPos;

    /// Current hand of the arm position
    Data::PrimitiveTypeArray<double> cartesianHand;

    /// Current goal position
    Data::PrimitiveTypeArray<double> cartesianTarget;

    /// Current goal position
    Data::PrimitiveTypeArray<double> cartesianDiff;
    
    /// Current motor speed
    Data::PrimitiveTypeArray<double> dataMotorSpeed;

    /// converter used to covnert motorPos to cartesionPos
    armlearn::kinematics::Converter *converter;

    /// Score of the training
    double score = 0.0;

    /// Reward of the last action done
    double reward = 0.0;

    /// Number of actions done in the episode
    size_t nbActions = 0;
    
    int nbActionsInThreshold=0;

    bool isMoving=true;

    /// Maximum number of actions doable in an episode 
    int nbMaxActions;

    /// Limit for the creation of of random targets
    double currentMaxLimitTarget = 10000;

    /// Limit for the creation of of random Starting position
    double currentMaxLimitStartingPos = 10000;

    /// Counter used to know if the currents limits have to be upgraded
    uint16_t counterIterationUpgrade = 0;

    /// Initial starting position of the arm
    std::vector<uint16_t> initStartingPos = BACKHOE_POSITION;

    /// Current Starting position of the arm
    std::vector<uint16_t> *currentStartingPos;

    /// Vector with Starting positions in keys and Targets positions in values used for the training
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<double>*>> trainingTrajectories;

    /// Vector with Starting positions in keys and Targets positions in values used for the training validation
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<double>*>> trainingValidationTrajectories;

    /// Vector with Starting positions in keys and Targets positions in values used for the validation
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<double>*>> validationTrajectories;
        
    /// Current generation
    int generation = 0;

    /// Target currently used to move the arm.
    armlearn::Input<double> *currentTarget;


    /// Vector that store all the motors positions of an episode
    std::vector<std::vector<uint16_t>> allMotorPos;

    /// Vector that store all the informations of an episode
    std::vector<int32_t> vectorValidationInfos;

    /// vector that store all the informations of each episode
    std::vector<std::vector<int32_t>> allValidationInfos;

    /// Checkpoint to get the duration of an episode (for testing logs)
    std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;

    /// Vector that contain the indices of the trajectories and their best score. Used if trajectory deletion is activated
    std::vector<std::pair<int, double>> scoreTrajectories;

    /// True if gegelati is running else an other algorithm : SAC for now
    bool gegelatiRunning = true;

    /// Motor speed : Use only if trainingParams.actionSpeed is true. speed is in motorPoint/iteration
    std::vector<double> motorSpeed = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    /// Current range target to reach
    double currentRangeTarget = 5;

    /// True if this is validation, else false
    bool isValidation = true;

    /// Vector that register all the possible goal to seek for in cartesian coordonates
    std::vector<std::vector<double>> dataTarget;

    /// Vector containing base segments to avoid collision with
    std::vector<std::vector<double>> baseSegments = {
        {68, 90, 9, 9}, {68, 68, 9, 95}, {15, 68, 95, 95}, {15, 15, 95, 151},
        {-68, -90, 9, 9}, {-68, -68, 9, 95}, {-15, -68, 95, 95}, {-15, -15, 95, 151},
    };

    /// @brief memory of each motor position during an episode, used to detect cycles
    std::vector<std::vector<uint16_t>> memoryMotorPos;

    /// @brief indicate if the arm is cycling (only under Gegelati)
    bool isCycling = false;

public:

    /// @brief Do not know
    /// @return 
    armlearn::communication::AbstractController *iniController() {
        auto conv = new armlearn::kinematics::BasicCartesianConverter(); // Create kinematics calculator
        auto arbotix_sim = new armlearn::communication::NoWaitArmSimulator(
                armlearn::communication::none); // Create robot simulator

        armlearn::WidowXBuilder builder;
        builder.buildConverter(*conv);
        builder.buildController(*arbotix_sim);

        arbotix_sim->connect();

        arbotix_sim->changeSpeed(50);

        arbotix_sim->updateInfos();

        converter = conv;

        return arbotix_sim; // defines the parent DeviceLearner "device" attribute
    }

    /**
     * Constructor.
     *
     * \param[in] handServosTrained boolean controlling whether the 2 servos
     * controlling the hand of the robotic arm are trained.
     */
    ArmLearnWrapper(int nbMaxActions, TrainingParameters& params, bool gegelatiRunning, bool handServosTrained = false)
            : LearningEnvironment((handServosTrained) ? 13 : 9), gegelatiRunning(gegelatiRunning), handServosTrained(handServosTrained),
              nbMaxActions(nbMaxActions), motorPos(6), cartesianHand(3), cartesianTarget(3), cartesianDiff(3), dataMotorSpeed((params.actionSpeed) ? 4:0),
              trainingTrajectories(), validationTrajectories(), trainingValidationTrajectories(),
              DeviceLearner(iniController()), params(params) {
        if(params.progressiveRangeTarget){
            this->currentRangeTarget = params.maxLengthTargets;
        } else {
            if (params.progressiveModeStartingPos) this->currentMaxLimitStartingPos=params.maxLengthStartingPos;
            if (params.progressiveModeTargets) this->currentMaxLimitTarget=params.maxLengthTargets;
            this->currentRangeTarget = params.rangeTarget;
        }

        rng.setSeed(params.seed);

        loadTargetCSV();
    }

    /**
    * \brief Copy constructor for the armLearnWrapper.
    */ 
    ArmLearnWrapper(const ArmLearnWrapper &other) : Learn::LearningEnvironment(other.nbActions), 
                                                    nbMaxActions(other.nbMaxActions), motorPos(other.motorPos), gegelatiRunning(other.gegelatiRunning),
                                                    cartesianHand(other.cartesianHand), cartesianTarget(other.cartesianTarget), cartesianDiff(other.cartesianDiff),
                                                    dataMotorSpeed(other.dataMotorSpeed),
                                                    trainingTrajectories(other.trainingTrajectories),
                                                    validationTrajectories(other.validationTrajectories),
                                                    trainingValidationTrajectories(other.trainingValidationTrajectories),
                                                    DeviceLearner(iniController()), params(other.params) {
        if(params.progressiveRangeTarget){
            this->currentRangeTarget = other.currentRangeTarget;
        } else {
            this->currentRangeTarget = params.rangeTarget;
        }               
    }

    /// @brief Destructor
    ~ArmLearnWrapper() {
        delete this->device;
        delete this->converter;
    };


    /// @brief Inherited via LearningEnvironment
    void doAction(uint64_t actionID) override;

    /// Do a multi continuous action.
    void doActionContinuous(std::vector<float> actions);

    /// execute the action choosed
    void executeAction(std::vector<double> motorAction);

    /// Update the memory of motor position and detect if gegelati is in a cycle
    void updateAndCheckCycles();

    /// Save the current motor position(
    void saveMotorPos();

    /// @brief Inherited via LearningEnvironment
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING, uint16_t iterationNumber = 0, uint64_t generationNumber = 0) override;

    /**
     * @brief Inherited via LearningEnvironment, create the state vector
     */ 
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

    /// @brief Clear a given proportion of the current set of training targets 
    void clearPropTrainingTrajectories();

    // Add the indices and scoresgiven to the scoreTrajectories vector
    void addToScoreTrajectories(int index, double score);


    ///@brief Inherited from LearningEnvironment.
    double getScore() const override;

    /// @brief Return the reward for the last action done;
    double getReward() const;

    /// @brief Inherited via LearningEnvironment
    bool isTerminal() const override;

    /// @brief Inherited via LearningEnvironment
    bool isCopyable() const override;

    /// @brief Inherited via LearningEnvironment
    virtual LearningEnvironment *clone() const;

    /** 
     * @brief Update the vector of training trajectories, which mean deleting a proportion of trajectories,
     * Then create an amount of Starting positions, that can be random of a predifined one.
     * Finally one target is created for each random starting position
     * 
     * @param[in] nbTrajectories Number of trajectories to create in total
     */ 
    void updateTrainingTrajectories(int nbTrajectories);

    /** 
     * @brief Update the vector of training validation trajectories, which mean deleting the trajectories,
     * Then create an amount of Starting positions, that can be random of a predifined one.
     * Finally one target is created for each random starting position
     * 
     * @param[in] nbTrajectories Number of trajectories to create in total
     */ 
    void updateTrainingValidationTrajectories(int nbTrajectories);

    /** 
     * @brief Update the vector of validation trajectories, which mean deleting all current trajectories,
     * Then create an amount of Starting positions, that are the predifined one.
     * Finally one target is created for each starting position
     * 
     * @param[in] nbTrajectories Number of trajectories to create in total
     */ 
    void updateValidationTrajectories(int nbTrajectories);

    /**
     * @brief Create and return a random position with the motors
     */
    std::vector<uint16_t> randomMotorPos(std::vector<double> cartesianGoal, bool validation, bool isTarget);

    /**
     * @brief Create and return a random starting position for the arm

     * @param[in] coefSize coefficient between 0 and 1 suse to be check that the random starting position generated as a distance bigger than coefSize * limit
     * @param[in] validation true if the target is for the validation, else false
     */
    std::vector<uint16_t>* randomStartingPos(bool validation);

    /**
     * @brief Create and return a random targets in cartesian coordonates
     * 
     * @param[in] startingPos starting position : used to calculate the distance to browse by the arm
     * @param[in] coefSize coefficient between 0 and 1 suse to be check that the random starting position generated as a distance bigger than coefSize * limit
     * @param[in] validation true if the target is for the validation, else false
     * @param[in] maxLength distance max that the arm will have to browse in the trajectory
     */
    armlearn::Input<double>* randomGoal(std::vector<uint16_t> startingPos, bool validation);

    /**
     * @brief Load the CSV containing equilibrate random positions
     */
    void loadTargetCSV();

    /**
     * @brief Puts a custom goal in the first slot of the trainingTargets list.
     */ 
    void customTrajectory(armlearn::Input<double> *newGoal, std::vector<uint16_t> startingPos, bool validation = false);

    /**
     * @brief Check if the current limits for starting position and targets have to be updated based on the bestResult
     * If the limits are updated, the trajectories are updated based on the nbIterationsPerPolicyEvaluation 
     */ 
    bool updateCurrentLimits(double bestResult, int nbIterationsPerPolicyEvaluation);

    /**
     * @brief Returns a string logging the goal (to use e.g. when there is a goal change)
     */ 
    std::string newGoalToString() const;

    /**
     * @brief Used to print the current situation (positions of the motors)
     */ 
    std::string toString() const override;

    /**
    * @brief Executes a learning algorithm on the learning set
    */
    virtual void learn() override {}

    /**
     * @brief Inherited via DeviceLearner
     */ 
    virtual void test() override {}

    /// Save the validation trajectories in a ValidationTrajectories.txt file
    void saveValidationTrajectories();

    /// Load the validation trajectories from a ValidationTrajectories.txt file
    void loadValidationTrajectories();

    // Log the trajectories store in allValidationInfos vector
    void logTestingTrajectories(bool usingGegelati);

    /**
     * @brief Inherited via DeviceLearner
     */ 
    virtual armlearn::Output<std::vector<uint16_t>> *produce(const armlearn::Input<uint16_t> &input) override {
        return new armlearn::Output<std::vector<uint16_t>>(std::vector<std::vector<uint16_t>>());
    }
    /**
     * @brief Return current motors position in a vector
     */ 
    std::vector<uint16_t> getMotorsPos();


    /**
     * @brief Set generation
     */ 
    void setgeneration(int newGeneration);

    /**
     * @brief Get the initStartingPos
     */ 
    std::vector<uint16_t> getInitStartingPos();

    /**
     * @brief Set the new init starting position
     */ 
    void setInitStartingPos(std::vector<uint16_t> newInitStartingPos);

    /// Get currentMaxLimitTarget
    double getCurrentMaxLimitTarget();

    /// Get currentMaxLimitStartingPos
    double getCurrentMaxLimitStartingPos();

    /// Get currentRangeTarget
    double getCurrentRangeTarget();

    /// Get distance from the arm to the target
    double getDistance();


    /**
     * @brief Return True if one of the motor has collision or is bellow 0 on z axis
     * 
     * @param[in] newMotorPos Position in motor point of the different motors
     */
    bool motorCollision(std::vector<uint16_t> newMotorPos);

    bool hasCollision(std::vector<double> armSegment, std::vector<double> baseSegment);


};

#endif
