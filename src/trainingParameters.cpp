#include <fstream>
#include <iostream>
#include <json.h>

#include "trainingParameters.h"

void TrainingParameters::readConfigFile(const char* path, Json::Value& root)
{
    std::ifstream ifs;
    ifs.open(path);

    if (!ifs.is_open()) {
        std::cerr << "Error : specified param file doesn't exist : " << path << std::endl;
        std::cerr << "\033[1;31mmake sure you are not executing from a different directory than the one containing the params folder.\033[0m" << std::endl;
        throw Json::Exception("aborting");
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = true;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        std::cout << errs << std::endl;
        std::cerr << "Ignoring ill-formed config file " << path << std::endl;
    }
}

void TrainingParameters::setAllParamsFrom(const Json::Value& root)
{
    for (std::string const& key : root.getMemberNames()) {
        if (root[key].size() == 0) {
            // we have a parameter without subtree (as a leaf)
            Json::Value value = root[key];
            setParameterFromString(key, value);
        }
    }
}

void TrainingParameters::setParameterFromString(const std::string& param, Json::Value const& value)
{
    if (param == "reachingObjectives"){
        reachingObjectives = (bool)value.asBool();
        return; 
    }
    if (param == "doTrainingValidation") {
        doTrainingValidation = (bool)value.asBool();
        return;
    }

    if (param == "progressiveRangeTarget") {
        progressiveRangeTarget = (bool)value.asBool();
        return;
    }

    if (param == "progressiveModeMotor") {
        // progressiveModeMotor always false if progressiveRangeTarget is true
        progressiveModeMotor = (progressiveRangeTarget) ? false: (bool)value.asBool();
        return;
    }


    if (param == "doRandomStartingPosition") {
        doRandomStartingPosition = (bool)value.asBool();
        return;
    }

    if (param == "progressiveModeTargets") {
        progressiveModeTargets = (bool)value.asBool();
        return;
    }

    if (param == "maxLengthTargets") {
        maxLengthTargets = (double)value.asDouble();
        return;
    }

    if (param == "progressiveModeStartingPos") {
        progressiveModeStartingPos = (bool)value.asBool();
        return;
    }

    if (param == "maxLengthStartingPos") {
        maxLengthStartingPos = (double)value.asDouble();
        return;
    }

    if (param == "coefficientUpgradeMult") {
        coefficientUpgradeMult = (double)value.asDouble();
        return;
    }

    if (param == "coefficientUpgradeAdd") {
        coefficientUpgradeAdd = (double)value.asDouble();
        return;
    }

    if (param == "nbIterationsUpgrade") {
        nbIterationsUpgrade = (int)value.asUInt();
        return;
    }

    if (param == "rangeTarget") {
        rangeTarget = (double)value.asDouble();
        return;
    }

    if (param == "startPreviousTPG") {
        startPreviousTPG = (bool)value.asBool();
        return;
    }

    if (param == "namePreviousTPG") {
        namePreviousTPG = (std::string)value.asString();
        return;
    }

    if (param == "controlTrajectoriesDeletion"){
        controlTrajectoriesDeletion = (bool)value.asBool();
        return;
    }

    if (param == "propTrajectoriesReused"){
        propTrajectoriesReused = (double)value.asDouble();
        return;
    }

    if (param == "penaltyMoveUnavailable"){
        penaltyMoveUnavailable = (double)value.asDouble();
        return;
    }

    if (param == "penaltySpeed"){
        penaltySpeed = (double)value.asDouble();
        return;
    }


    if (param == "coefRewardMultiplication"){
        coefRewardMultiplication = (double)value.asDouble();
        return;
    }

    if (param == "loadValidationTrajectories"){
        loadValidationTrajectories = (bool)value.asBool();
        return;
    }

    if (param == "saveValidationTrajectories"){
        saveValidationTrajectories = (bool)value.asBool();
        return;
    }

    if (param == "seed"){
        seed = (uint64_t)value.asUInt64();
        return;
    }

    if (param == "interactiveMode"){
        interactiveMode = (bool)value.asBool();
        return;
    }

    if (param == "sizeAction"){
        sizeAction = (double)value.asDouble();
        return;
    }

    if (param == "bonusNbIteration"){
        bonusNbIteration = (bool)value.asBool();
        return;
    }

    if (param == "meanScoreWithStd"){
        meanScoreWithStd = (bool)value.asBool();
        return;
    }

    if (param == "testing"){
        testing = (bool)value.asBool();
        return;
    }

    if (param == "testPath") {
        testPath = (std::string)value.asString();
        return;
    }

    if (param == "useInstrTrig") {
        useInstrTrig = (bool)value.asBool();
        return;
    }

    if (param == "useInstrLogExp") {
        useInstrLogExp = (bool)value.asBool();
        return;
    }

    if (param == "useInstrComparison") {
        useInstrComparison = (bool)value.asBool();
        return;
    }

    if (param == "useInstrExpensiveArithmetic") {
        useInstrExpensiveArithmetic = (bool)value.asBool();
        return;
    }

    if (param == "instrType") {
        std::string type = value.asString();
        if (type == "int" || type == "float" || type == "double" || type == "fixed_point") {
            instrType = type;
        } else {
            std::cerr << "Unknown instruction type: " << type << std::endl;
        }
        return;
    }

    if (param == "actionSpeed"){
        actionSpeed = (bool)value.asBool();
        return;
    }

    if (param == "realSimulation"){
        realSimulation = (bool)value.asBool();
        return;
    }

    if (param == "nbIterationTraining"){
        nbIterationTraining = (uint64_t)value.asUInt64();
        return;
    }

    if (param == "timeMaxTraining"){
        timeMaxTraining = (uint64_t)value.asUInt64();
        return;
    }

    if (param == "nbIterationRootCanceled"){
        nbIterationRootCanceled = (uint64_t)value.asUInt64();
        return;
    }

    if (param == "canDo360"){
        canDo360 = (bool)value.asBool();
        return;
    }

    if (param == "killIfCollision"){
        killIfCollision = (bool)value.asBool();
        return;
    }

    if (param == "instrSetName"){
        // its a higher level parameter, we don't need to handle it here
        //instrSetName = (std::string)value.asString();
        return;
    }
    // we didn't recognize the symbol
    std::cerr << "Ignoring unknown parameter " << param << std::endl;
}

void TrainingParameters::loadParametersFromJson(const char* path)
{
    Json::Value root;
    readConfigFile(path, root);
    setAllParamsFrom(root);
}
