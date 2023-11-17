
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

    if (param == "cooefficientUpgrade") {
        cooefficientUpgrade = (double)value.asDouble();
        return;
    }

    if (param == "nbIterationsUpgrade") {
        nbIterationsUpgrade = (int)value.asUInt();
        return;
    }

    if (param == "thresholdUpgrade") {
        thresholdUpgrade = (double)value.asDouble();
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

    if (param == "propTrajectoriesReused"){
        propTrajectoriesReused = (double)value.asDouble();
        return;
    }

    if (param == "coefRewardNbIterations"){
        coefRewardNbIterations = (double)value.asDouble();
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
