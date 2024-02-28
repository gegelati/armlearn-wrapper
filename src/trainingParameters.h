
#ifndef TRAINING_PARAMETERS_H
#define TRAINING_PARAMETERS_H

#include <thread>
#include <iostream>

namespace Json {
    class Value;
}

class TrainingParameters {


private:
    /**
     * \brief Puts the parameters described in the derivative tree root in
     * given LearningParameters.
     *
     * Browses the JSON tree. If the node we're looking at is a leaf,
     * we call setParameterFromString. Otherwise, we browe its children to
     * follow the known parameters structure.
     *
     * \param[in] root JSON tree we will use to set parameters.
     * \param[out] params the LearningParameters being updated.
     */
    void setAllParamsFrom(const Json::Value& root);

    /**
     * \brief Reads a given json file and puts the derivative tree in root.
     *
     * Opens the file and calls the parseFromStream() method from JsonCpp
     * which handles all the parsing of the JSON file. It eventually returns
     * errors in a parameter, e.g. if the file does not respect JSON format.
     * In this case, the file is simply ignored and it is logged explicitly.
     * However, in case of JsonCpp internal errors, there can be exceptions,
     * as described in throws.
     *
     * \param[in] path path of the JSON file from which the parameters are
     *            read.
     * \param[out] root JSON tree we are going to build with the file.
     * \throws std::exception if json parser settings are not in their
     * right formats.
     */
    void readConfigFile(const char* path, Json::Value& root);

    /**
     * \brief Given a parameter name, sets its value in given
     * LearningParameters.
     *
     * To find the right parameter, the method contains a lot of if
     * statements, each of them finishing by a return. These statements
     * compare the given parameter name to known parameters names.
     * If a parameter is found, it casts value to the right type and sets
     * the given parameter to this value.
     * If no parameter was found, it simply ignores the input and logs it
     * explicitly.
     *
     * \param[in] param the name of the LearningParameters being updated.
     * \param[in] value the value we want to set the parameter to.
     */
    void setParameterFromString(const std::string& param,
                                Json::Value const& value);

public:
    /// Choose between stopping in or reaching the objectives
    bool reachingObjectives = false;

    /// True if training validation is used for random starting pos and random target pos
    bool doTrainingValidation = false;

    /// Use the progressive mode for choosing randomly the targets. If true, progressiveModeMotor is ignored
    bool progressiveRangeTarget = false;

    /// If True, progressive mode is done by increasing motor position, instead it is the euclidian distance that increase, is ignored if progressiveRangeTarget is true
    bool progressiveModeMotor = false;

    /// True if the starting positions are set randomly
    bool doRandomStartingPosition = false;

    /// True if the sphere will grow progressivly
    bool progressiveModeTargets = false;

    /// Init size of the sphere within which the target will be instantiate
    double maxLengthTargets = 30.0;

    /// True if the sphere will grow progressivly
    bool progressiveModeStartingPos = false;

    /// Init size of the sphere within which the target will be instantiate
    double maxLengthStartingPos = 30.0;

    /// Upgrade coefficient of the sphere with multiplication
    double coefficientUpgradeMult = 1.2;

    /// Upgrade coefficient of the sphere with addition
    double coefficientUpgradeAdd = 20;

    /// Number of consecuitive iterations before upgrading the sphere
    int nbIterationsUpgrade = 3;

    /// rangeTarget that the best TPG has to surpass to increment the upgrade coefficient
    double rangeTarget = -3.0;

    /// True to start with a previous TPG
    bool startPreviousTPG = false;

    /// Name of the file of previous TPG
    std::string namePreviousTPG = "";

    /// Proportion of targets reused at each generation
    double propTrajectoriesReused = 1;

    /// True to activate a control over the deletion of trajectories
    bool controlTrajectoriesDeletion = false;

    /// Value to penalize the algorithm if an unavailable action is taken : reward = reward - penalty
    double penaltyMoveUnavailable = 0;

	/// Penalty for moving a motor : the goal is to avoid to much speed unecessary
	double penaltySpeed = 0;

    /// Coefficient to multiply the reward with
    double coefRewardMultiplication = 1;

    /// true to load the validation trajectories
    bool loadValidationTrajectories = false;

    /// true to save the validation trajectories
    bool saveValidationTrajectories = false;

    /// Seed to init the algorithm
    uint64_t seed = 0;

    /// Set interactive mode or not (usefull for calcul machine)
    bool interactiveMode = false;

    /// Size in degree of a discrete action, and max size of a continuous action
    double sizeAction = 1;

    /// if true, selection for gegelati is based on score (sum of reward), else on distance
    bool isScoreResult = false;

    /// if true, deactivate the training and only logs results
	bool testing = false;

    /// path to store the testing output
    std::string testPath = "outLogs/";

	/// To use distance 2D instruction
	bool useInstrDist2d = false;

	/// To use distance 3D instruction
	bool useInstrDist3d = false;

	/// To use spherical coordonates instructions
	bool useInstrSphericalCoord = false;

    /// To use getPi instruction
	bool useInstrPi = false;

    /// If false, action change the motor position, if true action change the motor speed
    bool actionSpeed = false;

    /// True to stop coordonate/target bellow 0 on z axis and collision
    bool realSimulation = true;

    // Number of training iteration (nbIterationsPerPolicyEvaluation will be only for validation/training validation)
	uint64_t nbIterationTraining = 1;

    // Set a limit of time to train (in seconds), if 0 : no limit
	uint64_t timeMaxTraining = 0;



    /**
     * \brief Loads a given json file and fills the parameters it contains
     * in given LearningParameters.
     *
     * High level method that simply calls more complicated ones as follow :
     * - readConfigFile to get the derivative tree from a JSON file path
     * - setAllParamsFrom to set the parameters given the obtained tree.
     *
     * \param[in] path path of the JSON file from which the parameters are
     *            read.
     * \param[out] params the LearningParameters being updated.
     */
    void loadParametersFromJson(const char* path);

};

#endif
