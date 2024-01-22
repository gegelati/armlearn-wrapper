
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
    bool reachingObjectives = true;

    /// True if training validation is used for random starting pos and random target pos
    bool doTrainingValidation = true;

    /// If True, progressive mode is done by increasing motor position, instead it is the euclidian distance that increase
    bool progressiveModeMotor = true;

    /// True if the starting positions are set randomly
    bool doRandomStartingPosition = true;

    /// True if the sphere will grow progressivly
    bool progressiveModeTargets = true;

    /// Init size of the sphere within which the target will be instantiate
    double maxLengthTargets = 30.0;

    /// True if the sphere will grow progressivly
    bool progressiveModeStartingPos = true;

    /// Init size of the sphere within which the target will be instantiate
    double maxLengthStartingPos = 30.0;

    /// Upgrade coefficient of the sphere with multiplication
    double coefficientUpgradeMult = 1.2;

    /// Upgrade coefficient of the sphere with addition
    double coefficientUpgradeAdd = 20;

    /// Number of consecuitive iterations before upgrading the sphere
    int nbIterationsUpgrade = 3;

    /// Threshold that the best TPG has to surpass to increment the upgrade coefficient
    double thresholdUpgrade = -3.0;

    /// True to start with a previous TPG
    bool startPreviousTPG = false;

    /// Name of the file of previous TPG
    std::string namePreviousTPG = "";

    /// Proportion of targets reused at each generation
    double propTrajectoriesReused = 1;

    /// Proportion of targets reused at each generation
    double coefRewardNbIterations = 0;

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
