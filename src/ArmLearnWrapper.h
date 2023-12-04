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
    Data::PrimitiveTypeArray<double> cartesianPos;

    /// Current arm and goal distance vector
    Data::PrimitiveTypeArray<double> cartesianDif;

    /// converter used to covnert motorPos to cartesionPos
    armlearn::kinematics::Converter *converter;

    /// Score of the training
    double score = 0;

    /// Reward of the last action done
    double reward = 0;

    /// Number of actions done in the episode
    size_t nbActions = 0;

    int nbActionsInThreshold=0;

    /// Maximum number of actions doable in an episode 
    int nbMaxActions;

    /// Coefficient to add a malus on the reward corresponding to the number of iterations in the episode
    double coefRewardNbIterations = 0;

    /// Initial starting position of the arm
    std::vector<uint16_t> initStartingPos = BACKHOE_POSITION;

    /// Current Starting position of the arm
    std::vector<uint16_t> *currentStartingPos;

    /// Target currently used to move the arm.
    armlearn::Input<int16_t> *currentTarget;


    /// Iterator of the vactor of training trajectories
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>::iterator trainingIterator;

    /// Vector with Starting positions in keys and Targets positions in values used for the training
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>> trainingTrajectories;


    /// Iterator of the vactor of training validation trajectories
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>::iterator trainingValidationIterator;

    /// Vector with Starting positions in keys and Targets positions in values used for the training validation
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>> trainingValidationTrajectories;


    /// Iterator of the vactor of validation trajectories
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>::iterator validationIterator;

    /// Vector with Starting positions in keys and Targets positions in values used for the validation
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>> validationTrajectories;
        
    /// Current generation
    int generation = 0;

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
    ArmLearnWrapper(int nbMaxActions, double coefRewardNbIterations=0, bool handServosTrained = false)
            : LearningEnvironment((handServosTrained) ? 13 : 9), handServosTrained(handServosTrained),
              trainingIterator(), validationIterator(), trainingValidationIterator(),
              nbMaxActions(), coefRewardNbIterations(),
              motorPos(6), cartesianPos(3), cartesianDif(3),
              DeviceLearner(iniController()) {
        this->nbMaxActions = nbMaxActions;
        this->coefRewardNbIterations = coefRewardNbIterations;
    }

    /**
    * \brief Copy constructor for the armLearnWrapper.
    */
    ArmLearnWrapper(const ArmLearnWrapper &other) : Learn::LearningEnvironment(other.nbActions),
                                                    trainingIterator(other.trainingIterator),
                                                    validationIterator(other.validationIterator), 
                                                    trainingValidationIterator(other.trainingValidationIterator),
                                                    nbMaxActions(other.nbMaxActions),  coefRewardNbIterations(other.coefRewardNbIterations),
                                                    motorPos(other.motorPos), cartesianPos(other.cartesianPos),
                                                    cartesianDif(other.cartesianDif), DeviceLearner(iniController()) {

        this->reset(0);
        computeInput();
    }

    /// @brief Destructor
    ~ArmLearnWrapper() {
        delete this->device;
        delete this->converter;
    };


    /// @brief Inherited via LearningEnvironment
    void doAction(uint64_t actionID) override;


    /// @brief Inherited via LearningEnvironment
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

    /**
     * @brief Inherited via LearningEnvironment, create the state vector
     */ 
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

    /// @brief Clear a given proportion of the current set of training targets 
    void clearPropTrainingTrajectories(double prop, bool doRandomStartingPos);



    /**
     * @brief Inherited from LearningEnvironment.
     *
     */
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
     * @param[in] doRandomStartingPosition true if the starting position are random, else they are all set to the init one
     * @param[in] propTrajectoriesReused proportion of trajectories reused (not deleted)
     * @param[in] limitTargets distance max that the arm will have to browse in the trajectory
     * @param[in] limitStartingPos distance max that the starting position will be from the init one
     */ 
    void updateTrainingTrajectories(int nbTrajectories, bool doRandomStartingPosition, double propTrajectoriesReused,
                                    double limitTargets, double limitStartingPos);

    /** 
     * @brief Update the vector of training validation trajectories, which mean deleting the trajectories,
     * Then create an amount of Starting positions, that can be random of a predifined one.
     * Finally one target is created for each random starting position
     * 
     * @param[in] nbTrajectories Number of trajectories to create in total
     * @param[in] doRandomStartingPosition true if the starting position are random, else they are all set to the init one
     * @param[in] propTrajectoriesReused proportion of trajectories reused (not deleted)
     * @param[in] limitTargets distance max that the arm will have to browse in the trajectory
     * @param[in] limitStartingPos distance max that the starting position will be from the init one
     */ 
    void updateTrainingValidationTrajectories(int nbTrajectories, bool doRandomStartingPosition, double propTrajectoriesReused,
                                              double limitTargets, double limitStartingPos);

    /** 
     * @brief Update the vector of validation trajectories, which mean deleting all current trajectories,
     * Then create an amount of Starting positions, that are the predifined one.
     * Finally one target is created for each random starting position
     * 
     * @param[in] nbTrajectories Number of trajectories to create in total
     * @param[in] doRandomStartingPosition true if the starting position are random, else they are all set to the init one
     */ 
    void updateValidationTrajectories(int nbTrajectories, bool doRandomStartingPosition);

    /**
     * @brief Create and return a random position with the motors
     */
    std::vector<uint16_t> randomMotorPos();

    /**
     * @brief Create and return a random starting position for the arm

     * @param[in] coefSize coefficient between 0 and 1 suse to be check that the random starting position generated as a distance bigger than coefSize * limit
     * @param[in] validation true if the target is for the validation, else false
     * @param[in] maxLength distance max between the generated starting pos and the initStartingPos
     */
    std::vector<uint16_t>* randomStartingPos(double coefSize, bool validation, double maxLength=0);

    /**
     * @brief Create and return a random targets in cartesian coordonates
     * 
     * @param[in] startingPos starting position : used to calculate the distance to browse by the arm
     * @param[in] coefSize coefficient between 0 and 1 suse to be check that the random starting position generated as a distance bigger than coefSize * limit
     * @param[in] validation true if the target is for the validation, else false
     * @param[in] maxLength distance max that the arm will have to browse in the trajectory
     */
    armlearn::Input<int16_t>* randomGoal(std::vector<uint16_t> startingPos, double coefSize, bool validation, double maxLength=0);

    /**
     * @brief Puts a custom goal in the first slot of the trainingTargets list.
     */ 
    void customTrajectory(armlearn::Input<int16_t> *newGoal, std::vector<uint16_t> startingPos, bool validation = false);

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
};

#endif
