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

#include "../ppo/armPpoEngine.h"
#include "../ppo/ppoParameters.h"
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
    trainingParams.testing=true;

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }

    
    PPOParameters ppoParams;
    ppoParams.loadParametersFromJson((pathParams + "ppoParams.json").c_str());

    if(argc > 4){
        ppoParams.pathModel = argv[4];
    }

    Learn::LearningParameters gegelatiParams;
    File::ParametersParser::loadParametersFromJson((pathParams + "params.json").c_str(), gegelatiParams);


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams, true);


    //Creation of the Output stream on cout and on the file
    auto nameLogs = "garbage";
    std::ofstream file((trainingParams.pathLogs + nameLogs + ".ods").c_str(), std::ios::out);
    // Instantiate the softActorCritic engine
    ArmPPOEngine learningAgent(ppoParams, &armLearnEnv, file, trainingParams, gegelatiParams.maxNbActionsPerEval, 
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

    int counterLetGegelati = 0;
    int nbMilimeterChange = 10;
    int nbIterationGoBackGegelati = 100;
    int nbEpisodes = 0;
    double scoreOrig = 0;
    int nbActionsEp = 0;
    int nbActions = 0;
    bool tpgAction = true;
    double bestDistance = 0;
    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < 1){

        // Reset part
        if (armLearnEnv.isTerminal() || nbActionsEp == gegelatiParams.maxNbActionsPerEval || nbActions == 0){
            scoreOrig += (nbActions == 0) ? 0 : armLearnEnv.getDistance();
            nbEpisodes += (nbActions == 0) ? 0 : 1;
            nbActionsEp = 0;
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            tpgAction = true;
            bestDistance = armLearnEnv.getDistance();
            armLearnEnv.setGegelatiRunning(true);
        }

        // Do action with either TPGs, either SAC
        if(tpgAction){
    	    auto actionID = inferenceTPG();
            std::vector<std::uint64_t> actionsID = {(uint64_t)actionID};
            armLearnEnv.doActions(actionsID);
        } else {
    	    auto actionsID = learningAgent.doOneActionInference();
            armLearnEnv.doActionContinuous(actionsID);
            counterLetGegelati+=1;
            

            // If the counter reached the value, go back to Gegelati Inference
            if (counterLetGegelati == nbIterationGoBackGegelati){
                tpgAction = true;
                armLearnEnv.setGegelatiRunning(true);

            }
        }
        nbActionsEp++;
        nbActions++;

        // If current distance is lower, save new best distance
        if(armLearnEnv.getDistance() < bestDistance){
            bestDistance = armLearnEnv.getDistance();
        }


        if(tpgAction){

            // Else if current Distance is best distance plus the value to change, swap to DeepRL algorithm
            if (armLearnEnv.getDistance() > bestDistance + nbMilimeterChange){
                counterLetGegelati = 0;
                tpgAction = false;
                armLearnEnv.setGegelatiRunning(false); 
            
            // Else if TPGs stop but did not collide (so stop because of cycle or just action to stop), swap to Deep Rl
            } else if(!armLearnEnv.getIsMoving() && !armLearnEnv.getArmCollide()){
                counterLetGegelati = 0;
                tpgAction = false;
                armLearnEnv.setIsMoving(true);
                armLearnEnv.setGegelatiRunning(false);
                armLearnEnv.setTerminal(false);
            }
        }
    }
    scoreOrig /= 100;
    armLearnEnv.logTestingTrajectories(true);
    std::cout << "Total score: " << scoreOrig << std::endl;

}