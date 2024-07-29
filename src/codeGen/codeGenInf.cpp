#include <iostream>

extern "C" {
#include "externHeader.h"
#include "codeGenArmlearn.h"
	/// instantiate global variable used to communicate between the TPG and the environment
	double* in1;
    double* in2;
    double* in3;
    double* in4;
}

#include "../ArmLearnWrapper.h"
#include "../instructions.h"
#include "../trainingParameters.h"
#include <filesystem>


int main(int argc, char* argv[]) {
    std::cout << "Start Code Gen inference application." << std::endl;

    uint64_t seed = 0;
    if(argc > 1 && std::strcmp(argv[1], "default") != 0){
        seed = std::stoi(argv[1]);
    }

    std::string pathParams = "../params/";
    if(argc > 2){
        pathParams = argv[2];
    }

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson((pathParams + "trainParams.json").c_str());
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson((pathParams + "/params.json").c_str(), params);

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(params.maxNbActionsPerEval, trainingParams, true);

	/// fetch data in the environment
	auto dataSources = armLearnEnv.getDataSources();
	auto& st1 = dataSources.at(0).get();
	in1 = st1.getDataAt(typeid(double), 0).getSharedPointer<double>().get();

    auto& st2 = dataSources.at(1).get();
	in2 = st2.getDataAt(typeid(double), 0).getSharedPointer<double>().get();

    auto& st3 = dataSources.at(2).get();
	in3 = st3.getDataAt(typeid(double), 0).getSharedPointer<double>().get();

    auto& st4 = dataSources.at(3).get();
	in4 = st4.getDataAt(typeid(double), 0).getSharedPointer<double>().get();

    armLearnEnv.loadValidationTrajectories(trainingParams.pathValidationTrajectories);

    trainingParams.testing=true;

    armLearnEnv.setHybridMode(true);
    int nbEpisodes = 0;
    double scoreOrig = 0;
    int nbActionsEp = 0;
    int nbActions = 0;
    double time = 0;


    std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;

    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < params.nbIterationsPerPolicyEvaluation){
        if (armLearnEnv.isTerminal() || nbActionsEp == params.maxNbActionsPerEval || nbActions == 0){
            if(nbActions > 0){
                scoreOrig += armLearnEnv.getScore();
                armLearnEnv.setTimeEnv(time);
                armLearnEnv.saveMotorPos();
                nbEpisodes++;
                if(nbEpisodes == params.nbIterationsPerPolicyEvaluation){
                    break;
                }
            }
            nbActionsEp = 0;
                time = 0;
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            

        }
        checkpoint = std::make_shared<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
    	auto actionID = inferenceTPG();
        time += ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();

        std::vector<std::uint64_t> actionsID = {(uint64_t)actionID};
        armLearnEnv.doActions(actionsID);
        nbActionsEp++;
        nbActions++;
    }
    scoreOrig /= 100;
    armLearnEnv.logTestingTrajectories(true);
    std::cout << "Total score: " << scoreOrig << std::endl;



}