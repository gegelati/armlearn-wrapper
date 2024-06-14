
#include <fstream>
#include <iostream>
#include <json.h>

#include "ppoParameters.h"

void PPOParameters::readConfigFile(const char* path, Json::Value& root)
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

void PPOParameters::setAllParamsFrom(const Json::Value& root)
{
    for (std::string const& key : root.getMemberNames()) {
        if (root[key].size() == 0) {
            // we have a parameter without subtree (as a leaf)
            Json::Value value = root[key];
            setParameterFromString(key, value);
        }
    }
}

void PPOParameters::setParameterFromString(const std::string& param, Json::Value const& value)
{
    if (param == "lr") {
        lr = (double)value.asDouble();
        return;
    }

if (param == "clip") {
        clip = (double)value.asDouble();
        return;
    }

    if (param == "gamma") {
        gamma = (double)value.asDouble();
        return;
    }

    if (param == "lambda") {
        lambda = (double)value.asDouble();
        return;
    }

    if (param == "std") {
        std = (double)value.asDouble();
        return;
    }

    if (param == "beta") {
        beta = (double)value.asDouble();
        return;
    }

    if (param == "sizeBuffer") {
        sizeBuffer = (int)value.asUInt();
        return;
    }

    if (param == "batchSize") {
        batchSize = (int)value.asUInt();
        return;
    }

    if (param == "nbEpochs") {
        nbEpochs = (double)value.asUInt();
        return;
    }

    if (param == "sizeHL1") {
        sizeHL1 = (int)value.asUInt();
        return;
    }

    if (param == "sizeHL2") {
        sizeHL2 = (int)value.asUInt();
        return;
    }

    if (param == "standardize") {
        standardize = (bool)value.asBool();
        return;
    }


    if (param == "loadModels") {
        loadModels = (bool)value.asBool();
        return;
    }

    if (param == "pathModel") {
        pathModel = (std::string)value.asString();
        return;
    }

    if (param == "multipleActions") {
        multipleActions = (bool)value.asBool();
        return;
    }

    if (param == "continuousActions") {
        continuousActions = (bool)value.asBool();
        return;
    }

    if (param == "falseSingleAction") {
        falseSingleAction = (bool)value.asBool();
        return;
    }

    // we didn't recognize the symbol
    std::cerr << "Ignoring unknown parameter " << param << std::endl;
}

void PPOParameters::loadParametersFromJson(const char* path)
{
    Json::Value root;
    readConfigFile(path, root);

    setAllParamsFrom(root);
}
