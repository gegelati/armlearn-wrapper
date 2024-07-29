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
    trainingParams.testing=true;

    if(argc > 3){
        trainingParams.pathLogs = argv[3];
    }

    
    SACParameters sacParams;
    sacParams.loadParametersFromJson((pathParams + "sacParams.json").c_str());

    if(argc > 4) {
        sacParams.pathModel = argv[4];
    }

    sacParams.loadModels = true;

    Learn::LearningParameters gegelatiParams;
    File::ParametersParser::loadParametersFromJson((pathParams + "params.json").c_str(), gegelatiParams);


    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(gegelatiParams.maxNbActionsPerEval, trainingParams, true);


    armLearnEnv.setHybridMode(true);

    
    // Set and Prompt the number of threads
    torch::set_num_threads(gegelatiParams.nbThreads);
    std::cout << "Number of threads: " << torch::get_num_threads() << std::endl;

    //Creation of the Output stream on cout and on the file
    auto nameLogs = "garbage";
    std::ofstream file((trainingParams.pathLogs + nameLogs + ".ods").c_str(), std::ios::out);
    // Instantiate the softActorCritic engine
    ArmSacEngine learningAgent(sacParams, &armLearnEnv, file, trainingParams, gegelatiParams.maxNbActionsPerEval, 
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
    int nbMilimeterChange = std::stoi(argv[5]);
    int nbIterationGoBackGegelati = std::stoi(argv[6]);

    std::cout<<nbMilimeterChange<<"-"<<nbIterationGoBackGegelati<<std::endl;

    int nbEpisodes = 0;
    double scoreOrig = 0;
    int nbActionsEp = 0;
    int nbActions = 0;
    bool tpgAction = true;
    double bestDistance = 0;

    double time = 0;
    int actBySac = 0;
    std::shared_ptr<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>> checkpoint;

    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < gegelatiParams.nbIterationsPerPolicyEvaluation){


        // Reset part
        if (armLearnEnv.isTerminal() || nbActionsEp == gegelatiParams.maxNbActionsPerEval || nbActions == 0){
            if(nbActions > 0){
                scoreOrig += armLearnEnv.getDistance();
                nbEpisodes++;
                armLearnEnv.setTimeEnv(time);
                armLearnEnv.saveMotorPos();
            }
            if(nbEpisodes == gegelatiParams.nbIterationsPerPolicyEvaluation){
                break;
            }
            nbActionsEp = 0;        
            time = 0;

            tpgAction = true;
            armLearnEnv.setGegelatiRunning(true);

            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            bestDistance = armLearnEnv.getDistance();

        }


        // Do action with either TPGs, either SAC
        if(tpgAction){

            checkpoint = std::make_shared<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
            auto actionID = inferenceTPG();
            time += ((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count();

            std::vector<std::uint64_t> actionsID = {(uint64_t)actionID};
            armLearnEnv.doActions(actionsID);
            nbActionsEp++;
            nbActions++;
        } else {
            
    	    auto results = learningAgent.doActionsInference(std::min((uint64_t)nbIterationGoBackGegelati, gegelatiParams.maxNbActionsPerEval - nbActionsEp));
            

            nbActionsEp+=results.first;
            nbActions+=results.first;

            time += results.second;

            tpgAction = true;
            armLearnEnv.setGegelatiRunning(true);
            bestDistance = armLearnEnv.getDistance(false);

        }

        


        // If current distance is lower, save new best distance
        bestDistance = std::min(armLearnEnv.getDistance(), bestDistance);

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
    scoreOrig /= gegelatiParams.nbIterationsPerPolicyEvaluation;

    
    armLearnEnv.setGegelatiRunning(true);
    armLearnEnv.logTestingTrajectories(true);
    std::cout << "Total score: " << scoreOrig << std::endl;

}