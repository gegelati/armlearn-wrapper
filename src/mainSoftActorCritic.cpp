#include <unordered_set>
#include <string>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <gegelati.h>
#include "trainingParameters.h"
#include "armLearnLogger.h"

#include "ArmLearnWrapper.h"
#include "softActorCritic/armSacEngine.h"
#include "softActorCritic/sacParameters.h"
#include <torch/torch.h>


int main(int argc, char* argv[]) {
    std::cout << "Start ArmLearner SAC application." << std::endl;

    uint64_t seed = 0;
    if(argc > 1 && std::strcmp(argv[1], "default") != 0){
        seed = std::stoi(argv[1]);
    }

    std::string pathParams = "params/";
    if(argc > 2){
        pathParams = argv[2];
    }

    // Set the parameters from Gegelati.
    // Loads them from "params.json" file
    Learn::LearningParameters gegelatiParams;
    File::ParametersParser::loadParametersFromJson((pathParams + "params.json").c_str(), gegelatiParams);

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((pathParams + "trainParams.json").c_str());

    SACParameters sacParams;
    sacParams.loadParametersFromJson((pathParams + "sacParams.json").c_str());

    if(argc > 3){
        sacParams.pathModel = argv[3];
    }

    if(argc > 4){
        trainingParams.pathLogs = argv[4];
    }

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams, false);
    armLearnEnv.loadTargetCSV(trainingParams.pathTargetCSV, seed);

    // Set and Prompt the number of threads
    torch::set_num_threads(gegelatiParams.nbThreads);

    // Set random seed
    torch::manual_seed(seed);
    std::cout << "Number of threads: " << torch::get_num_threads() << std::endl;


    // Generate validation targets.
    if(gegelatiParams.doValidation && !trainingParams.loadValidationTrajectories){
        armLearnEnv.updateValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }

    if(trainingParams.doTrainingValidation){
        // Update/Generate the first training validation trajectories
        armLearnEnv.updateTrainingValidationTrajectories(gegelatiParams.nbIterationsPerPolicyEvaluation);
    }



    //Creation of the Output stream on cout and on the file
    auto nameLogs = (!trainingParams.testing) ? "logsSAC" : "garbage";
    std::ofstream file((sacParams.pathModel + nameLogs + ".ods").c_str(), std::ios::out);

    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, file, trainingParams, gegelatiParams.maxNbActionsPerEval, 
                               gegelatiParams.doValidation);

    // Save the validation trajectories
    if (trainingParams.saveValidationTrajectories){
        armLearnEnv.saveValidationTrajectories(trainingParams.pathValidationTrajectories);
    }

    // Load the validation trajectories
    if(trainingParams.loadValidationTrajectories){
        armLearnEnv.loadValidationTrajectories(trainingParams.pathValidationTrajectories);
    }

    if(trainingParams.testing){
        learningAgent.testingModel(gegelatiParams.nbIterationsPerPolicyEvaluation);
    } else {

        std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint = std::make_shared<std::chrono::time_point<
        std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
        bool timeLimitReached = false;

        // Train for params.nbGenerations generations
        for (int i = 0; i < gegelatiParams.nbGenerations && !timeLimitReached; i++) {
            armLearnEnv.setgeneration(i);

            // Update/Generate the training trajectories
            armLearnEnv.updateTrainingTrajectories(trainingParams.nbIterationTraining);

            // Train
            learningAgent.trainOneGeneration(trainingParams.nbIterationTraining);

            // Does a validation or not according to the parameter doValidation
            if (gegelatiParams.doValidation){
                learningAgent.validateOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);
            }

            // Does a training validation or not according to doTrainingValidation
            if (trainingParams.doTrainingValidation) {
                learningAgent.validateTrainingOneGeneration(gegelatiParams.nbIterationsPerPolicyEvaluation);
            }

            learningAgent.logTimes();

            // Check time limit only if the parameter is above 0
            if(trainingParams.timeMaxTraining > 0){
                // Set true if the time is above the limit
                timeLimitReached = (((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count() > trainingParams.timeMaxTraining);
            }
    }


    }

    return 0;
}


