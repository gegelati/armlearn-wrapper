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

#include "../softActorCritic/armSacEngine.h"
#include "../softActorCritic/sacParameters.h"
#include <torch/torch.h>



int main(int argc, char* argv[]) {
    std::cout << "Start ArmLearner application." << std::endl;

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

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }

    
    SACParameters sacParams;
    sacParams.loadParametersFromJson((pathParams + "sacParams.json").c_str());



    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(1500, trainingParams, true);


    //Creation of the Output stream on cout and on the file
    auto nameLogs = "garbage";
    std::ofstream file((trainingParams.pathLogs + nameLogs + ".ods").c_str(), std::ios::out);
    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, file, trainingParams, 1500, 
                               true);


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


    int nbEpisodes = 0;
    double scoreOrig = 0;
    int nbActionsEp = 0;
    int nbActions = 0;
    bool tpgAction = true;
    double previousDistance = 0;
    int incr = 0;
    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < 100){
        if (armLearnEnv.isTerminal() || nbActionsEp == 1500 || nbActions == 0){
            scoreOrig += (nbActions == 0) ? 0 : armLearnEnv.getDistance();
            nbActionsEp = 0;
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            nbEpisodes++;
            tpgAction = true;
            previousDistance = armLearnEnv.getDistance();
            armLearnEnv.setGegelatiRunning(true);
            armLearnEnv.setIsMoving(true);
            incr = 0;
        }
        if(tpgAction){
    	    auto actionID = inferenceTPG();
            armLearnEnv.doAction(actionID);
        } else {
    	    auto actionID = learningAgent.doOneActionInference();
            armLearnEnv.doActionContinuous(actionID);
        }
        nbActionsEp++;
        nbActions++;


        if(tpgAction && armLearnEnv.getDistance() > previousDistance){
            incr++;
            if(incr == 5){
                tpgAction = false;
                armLearnEnv.setGegelatiRunning(false);    
            } else {
                incr = 0;
            }
            
        } else if(!armLearnEnv.getIsMoving() && tpgAction){
            armLearnEnv.setIsMoving(true);
            tpgAction = false;
            armLearnEnv.setGegelatiRunning(false);
            armLearnEnv.setTerminal(false);
            armLearnEnv.incrValKillCollision();
        }
        previousDistance = armLearnEnv.getDistance();
    }
    scoreOrig /= 100;
    armLearnEnv.logTestingTrajectories(true);
    std::cout << "Total score: " << scoreOrig << std::endl;

}