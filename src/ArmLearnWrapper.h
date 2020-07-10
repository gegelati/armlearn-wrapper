#ifndef TIC_TAC_TOE_WITH_OPPONENT_H
#define TIC_TAC_TOE_WITH_OPPONENT_H

#include <random>

#include <gegelati.h>
#include <armlearn/nowaitarmsimulator.h>
#include <armlearn/serialcontroller.h>
#include <armlearn/trajectory.h>

#include <armlearn/widowxbuilder.h>
#include <armlearn/optimcartesianconverter.h>
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
*/
class ArmLearnWrapper : public Learn::LearningEnvironment, armlearn::learning::DeviceLearner {
protected:
    void computeInput();

    double computeReward();

    bool terminal = false;

    /// Randomness control
    Mutator::RNG rng;

    /// Inputs of learning, positions to ask to the robot
    std::vector<armlearn::Input<uint16_t> *> targets;

    /// Current arm position
    Data::PrimitiveTypeArray<double> motorPos;

    /// Current arm position
    Data::PrimitiveTypeArray<double> cartesianPos;

    /// Current arm and goal distance vector
    Data::PrimitiveTypeArray<double> cartesianDif;

    armlearn::kinematics::Converter *converter;

    double score = 0;

    size_t nbActions = 0;

public:

    armlearn::communication::AbstractController *iniController() {
        auto conv = new armlearn::kinematics::OptimCartesianConverter(); // Create kinematics calculator
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
    */
    ArmLearnWrapper() // 1 motorPos/ motor = 6, 2 output directions / motor = 6
            : LearningEnvironment(13), targets(1), motorPos(6), cartesianPos(3), cartesianDif(3),
              DeviceLearner(iniController()) {

/*
        auto goal1 = new armlearn::Input<uint16_t>({0, 247, 267});
        auto goal2 = new armlearn::Input<uint16_t>({100, 347, 267});
        auto goal3 = new armlearn::Input<uint16_t>({0, 347, 167});
        //auto goal4 = new armlearn::Input<uint16_t>({8, 50, 150});
        targets.push_back(goal1);
        targets.push_back(goal2);
        targets.push_back(goal3);
        //targets.push_back(goal4);*/
    }

    armlearn::communication::AbstractController *generateControllerAndSetConverter() {

    }

/**
* \brief Copy constructor for the armLearnWrapper.
*
*/
    ArmLearnWrapper(const ArmLearnWrapper &other) : Learn::LearningEnvironment(other.nbActions), targets(other.targets),
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

/// Changes the goal, putting the first of the vector to the end
    void swapGoal(int i);

/// Generation a new  random
    armlearn::Input<uint16_t> *randomGoal();

/// Gives a custom goal to the environment
    void customGoal(armlearn::Input<uint16_t> *newGoal);

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

};

#endif