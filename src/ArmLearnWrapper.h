#ifndef TIC_TAC_TOE_WITH_OPPONENT_H
#define TIC_TAC_TOE_WITH_OPPONENT_H

#include <random>

#include <gegelati.h>
#include <armlearn/nowaitarmsimulator.h>
#include <armlearn/serialcontroller.h>
#include <armlearn/trajectory.h>

#include <armlearn/widowxbuilder.h>
#include <armlearn/basiccartesianconverter.h>
#include <armlearn/devicelearner.h>

// Proportion of target error in the reward
#define TARGET_PROP 0.7
// Coefficient of target error (difference between the real output and the target output, to minimize) when computing error between input and output
#define TARGET_COEFF (TARGET_PROP / (2 * 613))
// Coefficient of movement error (distance browsed by servomotors to reach target, to minimize) when computing error between input and output
#define MOVEMENT_COEFF ((1 - TARGET_PROP) / 5537)
// Coefficient of valid position error (returned if position is not valid)
#define VALID_COEFF -1
// Margin of error allowing to stop learning if error is below the number
#define LEARN_ERROR_MARGIN 0.005
// Coefficient decreasing the reward for a state as the state is closer to the initial state
#define DECREASING_REWARD 0.99

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
    void computeInput();

    double computeReward();

    bool terminal = false;

    /// Boolean used to control whether the LE includes the 2 servos
    /// controlling the rotation of the hand and the fingers.
    bool handServosTrained;

    /// Randomness control
    Mutator::RNG rng;

    /// Current arm position
    Data::PrimitiveTypeArray<double> motorPos;

    /// Current arm position
    Data::PrimitiveTypeArray<double> cartesianPos;

    /// Current arm and goal distance vector
    Data::PrimitiveTypeArray<double> cartesianDif;

    armlearn::kinematics::Converter *converter;

    double score = 0;

    size_t nbActions = 0;

    std::vector<uint16_t> startingPos = BACKHOE_POSITION;

    /// Target currently used to move the arm.
    armlearn::Input<int16_t> *currentTarget;

public:

    /// Inputs of learning, positions to ask to the robot in TRAINING mode
    std::vector<armlearn::Input<int16_t> *> trainingTargets;
    /// Inputs of learning, positions to ask to the robot in VALIDATION mode
    std::vector<armlearn::Input<int16_t> *> validationTargets;

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
    ArmLearnWrapper(bool handServosTrained = false)
            : LearningEnvironment((handServosTrained) ? 13 : 9), handServosTrained(handServosTrained),
              trainingTargets(1), validationTargets(1),
              motorPos(6), cartesianPos(3), cartesianDif(3),
              DeviceLearner(iniController()) {
    }

    /**
    * \brief Copy constructor for the armLearnWrapper.
    *
    */
    ArmLearnWrapper(const ArmLearnWrapper &other) : Learn::LearningEnvironment(other.nbActions),
                                                    trainingTargets(other.trainingTargets),
                                                    validationTargets(other.validationTargets),
                                                    motorPos(other.motorPos), cartesianPos(other.cartesianPos),
                                                    cartesianDif(other.cartesianDif), DeviceLearner(iniController()) {

        this->reset(0);
        computeInput();
    }

    /// Destructor
    ~ArmLearnWrapper() {
        delete this->device;
        delete this->converter;
    };


    /// Inherited via LearningEnvironment
    void doAction(uint64_t actionID) override;


    /// Inherited via LearningEnvironment
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

    /// Inherited via LearningEnvironment
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

/**
* Inherited from LearningEnvironment.
*
* The score is the sum of scores obtained during the game
* after each move, a score increment depending of the proximity of the arm
* regarding its goal is done.
*/
    double getScore() const override;

/// Inherited via LearningEnvironment
    bool isTerminal() const override;

/// Inherited via LearningEnvironment
    bool isCopyable() const override;

/// Inherited via LearningEnvironment
    virtual LearningEnvironment *clone() const;

    /**
    * \brief Set the currentTarget to the next target.
    *
    * The next currentTarget is chosen within the trainingTargets or
    * the validationTargets lists, depending on the given modes.
    * A left rotation of the selected list is done, to iterate on the list
    * of targets automatically when calling the swapGoal method.
    *
    */
    void swapGoal(Learn::LearningMode mode);

    /// Generation a new  random
    armlearn::Input<int16_t> *randomGoal(std::vector<std::string> tpara);

    /// Puts a custom goal in the first slot of the trainingTargets list.
    void customGoal(armlearn::Input<int16_t> *newGoal);

/// Returns a string logging the goal (to use e.g. when there is a goal change)
    std::string newGoalToString() const;

/// Used to print the current situation (positions of the motors)
    std::string toString() const override;

    /**
    * @brief Executes a learning algorithm on the learning set
    *
    */
    virtual void learn() override {}

    /// Inherited via DeviceLearner
    virtual void test() override {}

    /// Inherited via DeviceLearner
    virtual armlearn::Output<std::vector<uint16_t>> *produce(const armlearn::Input<uint16_t> &input) override {
        return new armlearn::Output<std::vector<uint16_t>>(std::vector<std::vector<uint16_t>>());
    }

    /// returns current motors position in a vector
    std::vector<uint16_t> getMotorsPos();

    /// sets a new position to begin simulation from
    void changeStartingPos(std::vector<uint16_t> newStartingPos);

};

#endif
