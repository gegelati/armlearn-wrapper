
#include <fstream>
#include <iostream>
#include <json.h>

#include "trainingParameters.h"

void TrainingParameters::readConfigFile(const char* path, Json::Value& root)
{
    std::ifstream ifs;
    ifs.open(path);

    if (!ifs.is_open()) {
        std::cerr << "Error : specified param file doesn't exist : " << path
                  << std::endl;
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

    if (param == "isScoreResult"){
        isScoreResult = (bool)value.asBool();
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

    if (param == "useInstrDist2d"){
        useInstrDist2d = (bool)value.asBool();
        return;
    }
    
    if (param == "useInstrDist3d"){
        useInstrDist3d = (bool)value.asBool();
        return;
    }
    
    if (param == "useInstrSphericalCoord"){
        useInstrSphericalCoord = (bool)value.asBool();
        return;
    }

    if (param == "actionSpeed"){
        actionSpeed = (bool)value.asBool();
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
