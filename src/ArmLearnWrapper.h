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

    template<class R, class T, class U>
    double computeReward(const std::vector<R> &target, const std::vector<T> &motorPos,
                         const std::vector<U> &cartesianPos) const;


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
            : LearningEnvironment(13), targets(), motorPos(6), cartesianPos(3), cartesianDif(3),
              DeviceLearner(iniController()) {


        auto goal = new armlearn::Input<uint16_t>({5, 50, 300});
        targets.push_back(goal);

        this->reset(0);
        computeInput();
    }

    armlearn::communication::AbstractController *generateControllerAndSetConverter() {

    }

/**
* \brief Copy constructor for the armLearnWrapper.
*
*/
    ArmLearnWrapper(const ArmLearnWrapper &other) : Learn::LearningEnvironment(other.nbActions),
                                                    motorPos(other.motorPos), cartesianPos(other.cartesianPos),
                                                    cartesianDif(other.cartesianDif), DeviceLearner(iniController()) {
        /*device->setPosition({(uint16_t)*(this->motorPos.getDataAt(typeid(double), 0).getSharedPointer<const double>()),
                             (uint16_t)*(this->motorPos.getDataAt(typeid(double), 1).getSharedPointer<const double>()),
                             (uint16_t)*(this->motorPos.getDataAt(typeid(double), 2).getSharedPointer<const double>()),
                             (uint16_t)*(this->motorPos.getDataAt(typeid(double), 3).getSharedPointer<const double>()),
                             (uint16_t)*(this->motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>()),
                             (uint16_t)*(this->motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>())});*/

        auto goal = new armlearn::Input<uint16_t>({5, 50, 300});
        targets.push_back(goal);

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

/// Used to print the current situation (positions of the motors)
    std::string toString() const override;

    /**
 * @brief Executes a learning algorithm on the learning set --- UNUSED AND USELESS TODO fix that
 *
 */
    virtual void learn() override {}

    /**
* @brief Tests the efficiency of the learning  --- UNUSED AND USELESS TODO fix that
*
*/
    virtual void test() override {}

    // UNUSED AND USELESS TODO fix that
    virtual armlearn::Output<std::vector<uint16_t>> *produce(const armlearn::Input<uint16_t> &input) override {
        return new armlearn::Output<std::vector<uint16_t>>(std::vector<std::vector<uint16_t>>());
    }

};

#endif