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


int main() {

    TrainingParameters trainingParams;
    trainingParams.loadParametersFromJson("params/trainParams.json");
    trainingParams.testing = true;

    // Instantiate the LearningEnvironment
    ArmLearnWrapper armLearnEnv(1500, trainingParams, true);

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

    armLearnEnv.loadValidationTrajectories();

    int nbEpisodes = 0;
    double scoreOrig = 0;
    int nbActionsEp = 0;
    int nbActions = 0;
    std::cout << "Play with TPG code" << std::endl;
    while(nbEpisodes < 100){
        if (armLearnEnv.isTerminal() || nbActionsEp == 1500 || nbActions == 0){
            scoreOrig += (nbActions == 0) ? 0 : armLearnEnv.getScore();
            nbActionsEp = 0;
            armLearnEnv.reset(nbActions, Learn::LearningMode::VALIDATION, nbEpisodes, 0);
            nbEpisodes++;
        }
    	double actionID = -1;
        inferenceTPG(&actionID);
        armLearnEnv.doAction(actionID);
        nbActionsEp++;
        nbActions++;
    }
    scoreOrig /= 100;
    constexpr bool USING_GEGELATI = true; 
    armLearnEnv.logTestingTrajectories(USING_GEGELATI, "outLogs/CodeGen");
    std::cout << "Total score: " << scoreOrig << std::endl;
}