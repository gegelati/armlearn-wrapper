#include "ArmLearnWrapper.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>

void ArmLearnWrapper::computeInput() {

    // Get the device state
    auto deviceState = DeviceLearner::getDeviceState();

    std::vector<uint16_t> newMotorPos;
    int indInput = 0;

    // For each motor and each value of motor (here value is only position)
    for (auto &motorState : deviceState) {
        for (unsigned short &value : motorState) {
            // Get the value
            motorPos.setDataAt(typeid(double), indInput, value);
            newMotorPos.emplace_back(value);
            indInput++;
        }
    }

    // Get the cartesian coordonates of the motors
    auto newCartesianCoords = converter->computeServoToCoord(newMotorPos)->getCoord();
   
    // For each motor, save the position and the relative position with the target
    for (int i = 0; i < newCartesianCoords.size(); i++) {
        cartesianHand.setDataAt(typeid(double), i, newCartesianCoords[i]);
        cartesianTarget.setDataAt(typeid(double), i, this->currentTarget->getInput()[i]);
        cartesianDiff.setDataAt(typeid(double), i, this->currentTarget->getInput()[i] - newCartesianCoords[i]);
    }
}

std::vector<std::reference_wrapper<const Data::DataHandler>> ArmLearnWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.emplace_back(cartesianTarget);
    result.emplace_back(cartesianHand);
    result.emplace_back(cartesianDiff);
    result.emplace_back(motorPos);
    return result;
}

void ArmLearnWrapper::doAction(uint64_t actionID) {

    std::vector<double> out;
    //double step = M_PI / 180 * params.sizeAction; // discrete rotations of 1°
    double step  = params.sizeAction;

    // Get the action
    switch (actionID) {
        case 0:
            out = {step, 0, 0, 0, 0, 0};
            break;
        case 1:
            out = {0, step, 0, 0, 0, 0};
            break;
        case 2:
            out = {0, 0, step, 0, 0, 0};
            break;
        case 3:
            out = {0, 0, 0, step, 0, 0};
            break;
        case 4:
            out = {-step, 0, 0, 0, 0, 0};
            break;
        case 5:
            out = {0, -step, 0, 0, 0, 0};
            break;
        case 6:
            out = {0, 0, -step, 0, 0, 0};
            break;
        case 7:
            out = {0, 0, 0, -step, 0, 0};
            break;
        case 8:
            out = {0, 0, 0, 0, 0, 0};
            isMoving=false;
            break;

            // Following cases only when the hand is trained
        case 9:
            out = {0, 0, 0, 0, step, 0};
            break;
        case 10:
            out = {0, 0, 0, 0, 0, step};
            break;
        case 11:
            out = {0, 0, 0, 0, -step, 0};
            break;
        case 12:
            out = {0, 0, 0, 0, 0, -step};
            break;
    }

    // Scale the positions : this return a vector of int between 0 and 4096 corresponding to the step
    auto scaledOutput = device->scalePosition({0, 0, 0, 0, 0, 0}, -M_PI, M_PI);


    double inputI = (double) *(motorPos.getDataAt(typeid(double), 0).getSharedPointer<const double>());
    // Substract by 2048 to get the out value indacted by the action
    if(out[0] + inputI >= 2  && out[0] + inputI <= 4094){
        scaledOutput[0] = out[0] + inputI;
    } else {
        scaledOutput[0] = inputI;
        isMoving=false;
    }

    // changes relative coordinates to absolute
    for (int i = 1; i < 4; i++) {

        inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        // Substract by 2048 to get the out value indacted by the action
        if(out[i] + inputI >= 1025 && out[i] + inputI <= 3071){
            scaledOutput[i] = out[i]  + inputI;
        } else {
            scaledOutput[i] = inputI;
            isMoving=false;
        }

    }

    inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
    scaledOutput[4] = (scaledOutput[4] - 511) + inputI;
    inputI = (double) *(motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>());
    scaledOutput[5] = (scaledOutput[5] - 256) + inputI;

    auto validOutput = device->toValidPosition(scaledOutput);
    device->setPosition(validOutput); // Update position
    device->waitFeedback();

    
    computeInput(); // to update  positions

    nbActions++;
    reward = computeReward(); // Computation of reward
    score += reward;

    
    if(params.testing){
        allMotorPos.push_back(getMotorsPos());
        if(terminal || nbActions == nbMaxActions){
            vectorValidationInfos.push_back(static_cast<int32_t>(getScore()));
            vectorValidationInfos.push_back(static_cast<int32_t>(nbActions));
            for(auto motor_value: allMotorPos){
                vectorValidationInfos.push_back(motor_value[0]);
                vectorValidationInfos.push_back(motor_value[1]);
                vectorValidationInfos.push_back(motor_value[2]);
                vectorValidationInfos.push_back(motor_value[3]);
            }
            allValidationInfos.push_back(vectorValidationInfos);
        }
    }

}

void ArmLearnWrapper::doActionContinuous(std::vector<float> actions) {


    std::vector<double> out;
    for (float &action : actions) {
        out.push_back(round(params.sizeAction * action));
    }
    out.push_back(0.0);
    out.push_back(0.0);


    

    // Scale the positions : this return a vector of int between 0 and 4096 corresponding to the step
    auto scaledOutput = device->scalePosition({0, 0, 0, 0, 0, 0}, -M_PI, M_PI);


    double inputI = (double) *(motorPos.getDataAt(typeid(double), 0).getSharedPointer<const double>());
    // Substract by 2048 to get the out value indacted by the action
    if(out[0] + inputI >= 2  && out[0] + inputI <= 4094){
        scaledOutput[0] = out[0] + inputI;
    } else {
        scaledOutput[0] = inputI;
    }

    // changes relative coordinates to absolute
    for (int i = 1; i < 4; i++) {

        inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        // Substract by 2048 to get the out value indacted by the action
        if(out[i] + inputI >= 1025 && out[i] + inputI <= 3071){
            scaledOutput[i] = out[i]  + inputI;
        } else {
            scaledOutput[i] = inputI;
        }

    }



    inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
    scaledOutput[4] = (scaledOutput[4] - 511) + inputI;
    inputI = (double) *(motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>());
    scaledOutput[5] = (scaledOutput[5] - 256) + inputI;

    auto validOutput = device->toValidPosition(scaledOutput);
    device->setPosition(validOutput); // Update position
    device->waitFeedback();

    
    computeInput(); // to update  positions

    nbActions++;
    reward = computeReward(); // Computation of reward
    score += reward;
    
    if(params.testing){
        allMotorPos.push_back(getMotorsPos());
        if(terminal || nbActions == nbMaxActions){
            vectorValidationInfos.push_back(static_cast<int32_t>(getScore()));
            vectorValidationInfos.push_back(static_cast<int32_t>(nbActions));
            for(auto motor_value: allMotorPos){
                vectorValidationInfos.push_back(motor_value[0]);
                vectorValidationInfos.push_back(motor_value[1]);
                vectorValidationInfos.push_back(motor_value[2]);
                vectorValidationInfos.push_back(motor_value[3]);
            }
            allValidationInfos.push_back(vectorValidationInfos);
        }
    }


}


double ArmLearnWrapper::computeReward() {

    // Compute que Distance with the target
    auto err = getDistance();

    /// Compute the number of actions taken in the episode divide by the maximum number of actions takeable in an episode
    /// This ratio is multiplied by a coefficient that allow to choose the impact of this ratio on the reward
    //double valNbIterations = params.coefRewardNbIterations * (static_cast<double>(nbActions) / nbMaxActions);

    // Tempory reward to force to stop close to the objective
    if (err < params.thresholdUpgrade){
        // Incremente a counter
        nbActionsInThreshold++;

    // If not close to the objective
    } else{
        // reset counter
        nbActionsInThreshold=0;        
    }

    // If the counter reach 10 or terminal is true (because the arm can stop and set terminal=true with action 8)
    /*if(nbActionsInThreshold == 10 || !isMoving){
        terminal = true;
    }*/

    if(!isMoving){
        terminal = true;
    }

    if(params.reachingObjectives){
        if(err < params.thresholdUpgrade){
            terminal = true;
            return 1000;
        }
    } else if(nbActionsInThreshold == 10 || !isMoving){
        terminal = true;
        if(err < params.thresholdUpgrade){
            return 1000;
        }
    }

    // If the arm is not moving anymore, the reward is multiplied by the numper of action normally to come
    double penalty = 1;
    if(!isMoving){
        penalty = nbMaxActions - nbActions;
    }

    // Return distance divided by the initCurrentMaxLimitTarget (this will push the arm to stay in the initCurrentMaxLimitTarget)
    return -err * params.coefRewardMultiplication * penalty;
    
}


void ArmLearnWrapper::reset(size_t seed, Learn::LearningMode mode, uint16_t iterationNumber, uint64_t generationNumber) {

    // Get the right trajectories' map
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<double>*>>* trajectories;
    
    switch (mode) {
        case Learn::LearningMode::TRAINING:
            trajectories = &trainingTrajectories;
            break;
        case Learn::LearningMode::VALIDATION:
            trajectories = &validationTrajectories;
            break;
        case Learn::LearningMode::TESTING:
            trajectories = &trainingValidationTrajectories;
            break;
    }

    if(params.testing){
        allMotorPos.clear();
        vectorValidationInfos.clear();

        
        for(auto val: *trajectories->at(iterationNumber).first){
            vectorValidationInfos.push_back(val);
        }

        
        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[0]);
        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[1]);
        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[2]);
    }

    // Change the starting position
    this->currentStartingPos = trajectories->at(iterationNumber).first;

    device->setPosition(*currentStartingPos); // Reset position
    device->waitFeedback();

    // Change the target
    this->currentTarget = trajectories->at(iterationNumber).second;
    computeInput();

    // Init environnement parameters
    reward = 0;
    score = 0;
    nbActions = 0;
    terminal = false;
    nbActionsInThreshold=0;
    isMoving = true;
}


void ArmLearnWrapper::addToDeleteTraj(int index){
    trajToDelete.push_back(index);
}

void ArmLearnWrapper::deleteTrajectory(){

    for(int i = trainingTrajectories.size() - 1; i >= 0; --i) {

        auto testFind = std::find(trajToDelete.begin(), trajToDelete.end(), i);

        if (testFind != trajToDelete.end() || !params.controlTrajectoriesDeletion){
            auto iterToDelete = trainingTrajectories.begin();
            std::advance(iterToDelete, i);

            if (params.doRandomStartingPosition) delete iterToDelete->first;
            delete iterToDelete->second;

            trainingTrajectories.erase(iterToDelete);
        }


    }

    trajToDelete.clear();
}

void ArmLearnWrapper::clearPropTrainingTrajectories(){

    // Do not clear and just return if the vector is empty
    if (trainingTrajectories.size() == 0)
        return;

    // Clear all and return if the proportion is 1 (or above, even if it should not be higher than 1)
    if (params.propTrajectoriesReused >= 1){
        trainingTrajectories.clear();
        return;
    }

    // Compute the number of trajectories we want to delete
    auto nbDeletedTrajectories = static_cast<int>(round(trainingTrajectories.size() * (1-params.propTrajectoriesReused)));

    // Take an iterator to reach the last deleted trajectory
    auto it = trainingTrajectories.begin();
    std::advance(it, nbDeletedTrajectories);

    // Delete all the pointers from memory
    std::for_each(trainingTrajectories.begin(), it, [this](auto& pair){
         if (this->params.doRandomStartingPosition) delete pair.first; // check doublon pointeur
         delete pair.second;
    }); 
    // Delete then the pair in the vector
    trainingTrajectories.erase(trainingTrajectories.begin(), it);
}


double ArmLearnWrapper::getScore() const {
    return score;
}

double ArmLearnWrapper::getReward() const {
    return reward;
}

bool ArmLearnWrapper::isTerminal() const {
    return terminal;
}

bool ArmLearnWrapper::isCopyable() const {
    return true;
}


void ArmLearnWrapper::updateTrainingTrajectories(int nbTrajectories){

    deleteTrajectory();

    // Clear a define prortion of the training targets by giving the proportion of targets reused
    //clearPropTrainingTrajectories();

    while (trainingTrajectories.size() < nbTrajectories){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (params.doRandomStartingPosition) ? randomStartingPos(false) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, false);



        // add the pair startingPos and target to the vector
        trainingTrajectories.push_back(std::make_pair(newStartingPos, newTarget));

    }
}

void ArmLearnWrapper::updateTrainingValidationTrajectories(int nbTrajectories){


    // Clear a define prortion of the training targets by giving the proportion of targets reused
    std::for_each(trainingValidationTrajectories.begin(), trainingValidationTrajectories.end(), [this](auto& pair){
         if (this->params.doRandomStartingPosition) delete pair.first; // check doublon pointeur
         delete pair.second;
    }); 
    trainingValidationTrajectories.clear();

    for (int i=0; i<nbTrajectories; i++){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (params.doRandomStartingPosition) ? randomStartingPos(false) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, false);

        // add the pair startingPos and target to the vector
        trainingValidationTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }
}

void ArmLearnWrapper::updateValidationTrajectories(int nbTrajectories){

    // Clear all the current validation trajectories
    std::for_each(validationTrajectories.begin(), validationTrajectories.end(), [this](auto& pair){
         delete pair.second;
    }); 
    validationTrajectories.clear();

    for (int i=0; i<nbTrajectories; i++){

        // Get a new starting pos based on the BACKHOE_POSITION
        auto newStartingPos = &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, true);

        // add the pair startingPos and target to the vector
        validationTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }
}

std::vector<uint16_t> ArmLearnWrapper::randomMotorPos(bool validation, bool isTarget){
    uint16_t i, j, k, l;
    std::vector<uint16_t> newMotorPos, validMotorPos;

    auto limit = (isTarget) ? currentMaxLimitTarget: currentMaxLimitStartingPos;


    if(!validation && params.progressiveModeMotor){
        
        int16_t valueNeeded = 2048 % (int)params.sizeAction;
        i = (int16_t) (valueNeeded + (int)(rng.getInt32(std::max(2.0, 2048 - limit) - valueNeeded, std::min(4094.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
        j = (int16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
        k = (int16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
        l = (int16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
        
    }
    else{
        int16_t valueNeeded = 2048 % (int)params.sizeAction;
        i = (int16_t) (valueNeeded + (int)(rng.getInt32(1 - valueNeeded, 4096 - valueNeeded) / params.sizeAction) * params.sizeAction);
        j = (int16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
        k = (int16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
        l = (int16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
    }

    // Create the vector of motor positions
    newMotorPos = {i,j,k,l,512,256};

    // Use this function to convert the vector of motor positions into a valid one
    validMotorPos = device->toValidPosition(newMotorPos);

    return validMotorPos;
}

std::vector<uint16_t> *ArmLearnWrapper::randomStartingPos(bool validation){

    std::vector<uint16_t> motorPos;
    std::vector<double> newStartingPos;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    // Do one time then only while the distance is above the distance between the new starting position and the initial one
    do {
        // Get a random motor positions
        motorPos = randomMotorPos(validation, false);

        // Compute the cartesian coordonates of those motor positions
        newStartingPos = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance the new starting position and the initial one
        distance = computeSquaredError(converter->computeServoToCoord(initStartingPos)->getCoord(), newStartingPos);

    } while (!validation && distance > currentMaxLimitStartingPos && !params.progressiveModeMotor);

    return new std::vector<uint16_t>(motorPos);

}

armlearn::Input<double> *ArmLearnWrapper::randomGoal(std::vector<uint16_t> startingPos, bool validation){

    std::vector<uint16_t> motorPos;
    std::vector<double> newCartesianCoords;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    // Do one time then only while the distance is above the distance to browse
    do {
        // Get a random motor positions
        motorPos = randomMotorPos(validation, true);

        // Compute the cartesian coordonates of those motor positions
        newCartesianCoords = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance to browse
        distance = computeSquaredError(converter->computeServoToCoord(startingPos)->getCoord(), newCartesianCoords);
        
    } while (!validation && distance > currentMaxLimitTarget && !params.progressiveModeMotor);

    // Create the input to return
    return new armlearn::Input<double>(
    {
        (double) (newCartesianCoords[0]), //X
        (double) (newCartesianCoords[1]), //Y
        (double) (newCartesianCoords[2])}); //Z
}


void ArmLearnWrapper::customTrajectory(armlearn::Input<double> *newGoal, std::vector<uint16_t> startingPos, bool validation) {

    // Get the right vector of trajectories
    auto trajectories = (validation) ? trainingTrajectories : validationTrajectories;

    // Delete the first key/value pair if the vector is not empty
    if(trajectories.size() > 0){
        auto iterator = trajectories.begin();
        if (this->params.doRandomStartingPosition) delete iterator->first;
        delete iterator->second;
        trajectories.erase(iterator);
    }
    
    // Add the custom target with the corresponding starting position
    trajectories.push_back(std::make_pair(&startingPos, newGoal));
}

void ArmLearnWrapper::updateCurrentLimits(double bestResult, int nbIterationsPerPolicyEvaluation){
    // If the best TPG is above the threshold for upgrade
    if (bestResult < params.thresholdUpgrade){

        // Incremente the counter for upgrading the max current limit
        counterIterationUpgrade += 1;

        // If the counter reach the number of iterations to upgrade
        if(counterIterationUpgrade == params.nbIterationsUpgrade){

            // Upgrade the limit of tagets
            if (params.progressiveModeTargets){
                currentMaxLimitTarget = std::min(currentMaxLimitTarget * params.coefficientUpgradeMult, currentMaxLimitTarget + params.coefficientUpgradeAdd);
                currentMaxLimitTarget = std::min(currentMaxLimitTarget, 750.0d);
            }


            // Upgrade the limit of starting positions
            if (params.progressiveModeStartingPos){
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos * params.coefficientUpgradeMult, currentMaxLimitStartingPos + params.coefficientUpgradeAdd);
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos, 750.0d);
            }


            counterIterationUpgrade = 0;

            // Update the training validation trajectories
            updateTrainingValidationTrajectories(nbIterationsPerPolicyEvaluation);
            
        }
    }
    // Reset the counter
    else
        counterIterationUpgrade = 0;
}

std::string ArmLearnWrapper::newGoalToString() const {

    // Log the current coordonate of the target
    std::stringstream toLog;
    toLog << " - (new goal : ";
    toLog << this->currentTarget->getInput()[0] << " ; ";
    toLog << this->currentTarget->getInput()[1] << " ; ";
    toLog << this->currentTarget->getInput()[2] << " ; ";
    toLog << ")" << std::endl;
    return toLog.str();
}

std::string ArmLearnWrapper::toString() const {
    std::stringstream res;

    // Log the current position of the motor
    for (int i = 0; i < 6; i++) {
        double input = (double) *(this->motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        res << input << " ; ";
    }

    // Log the current coordonates of the arm in cartesian coords
    res << "    -->    ";
    for (int i = 0; i < 3; i++) {
        double input = (double) *(this->cartesianHand.getDataAt(typeid(double), i).getSharedPointer<const double>());
        res << input << " ; ";
    }

    // Log the target coordonate in cartesian coords
    res << " - (goal : ";
    res << this->currentTarget->getInput()[0] << " ; ";
    res << this->currentTarget->getInput()[1] << " ; ";
    res << this->currentTarget->getInput()[2] << " ; ";
    res << ")";

    return res.str();
}

Learn::LearningEnvironment *ArmLearnWrapper::clone() const {
    return new ArmLearnWrapper(*this);
}

void ArmLearnWrapper::saveValidationTrajectories() {
    // Create file
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";
    std::ofstream outFile((slashToAdd + "params/ValidationTrajectories.txt").c_str());

    if (outFile.is_open()) {
        // For each validation trajectories
        for (auto item : validationTrajectories) {
            // Get the starting position
            for (auto value : *item.first) {
                outFile << value << " ";
            }

            // Then the target
            outFile << item.second->getInput()[0] << " ";
            outFile << item.second->getInput()[1] << " ";
            outFile << item.second->getInput()[2] << " " << std::endl;;
        }
        outFile << std::endl;
        outFile.close();
    } else {
        std::cerr << "Error while openning file for saving validation trajectories" << std::endl;
    }
}

void ArmLearnWrapper::loadValidationTrajectories() {


    // Clear the trajectories
    std::for_each(validationTrajectories.begin(), validationTrajectories.end(), [this](auto& pair){
        delete pair.second;
    }); 
    validationTrajectories.clear();

    // Get file
    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";
    std::ifstream inFile((slashToAdd + "params/ValidationTrajectories.txt").c_str());

    if (inFile.is_open()) {
        int value;
        inFile >> value;
        do {
            std::vector<uint16_t>* startingPos = new std::vector<uint16_t>();
            int i = 0;
            do { // Get the starting position
                startingPos->push_back(static_cast<uint16_t>(value));
                i++;
            } while (inFile >> value && i < 6);

            std::vector<int> target;
            do { // Get the target
                target.push_back(static_cast<uint16_t>(value));
                i++;
            } while (inFile >> value && i < 9);

            auto targetInput = new armlearn::Input<double>({
                (int16_t) (target[0]), //X
                (int16_t) (target[1]), //Y
                (int16_t) (target[2])});

            // Add to trajectories
            validationTrajectories.push_back(std::make_pair(startingPos, targetInput));
        } while(inFile.peek() != EOF);

        // Close the file
        inFile.close();

    } else {
        std::cerr << "Error while openning file for loading validation trajectories" << std::endl;
    }
}

void ArmLearnWrapper::logTestingTrajectories(bool usingGegelati){

    // Nom du fichier CSV
    std::string fileName = (usingGegelati) ? "outLogs/outputGegelati.csv": "outLogs/outputSAC.csv";

    // Ouverture du fichier en mode écriture
    std::ofstream outputFile(fileName);

    // Vérification si le fichier est correctement ouvert
    if (outputFile.is_open()) {
        outputFile<<"armPos0,"<<"armPos1,"<<"armPos2,"<<"armPos3,"<<"armPos4,"<<"armPos5,";
        outputFile<<"targetPos0,"<<"targetPos1,"<<"targetPos2,"<<"Score,"<<"NbActions,"<<"MotorPos"<<std::endl;
        // Écriture des données dans le fichier CSV
        for (const auto &row : allValidationInfos) {
            for (size_t i = 0; i < row.size(); ++i) {
                outputFile << row[i];

                // Ajout d'une virgule sauf pour le dernier élément
                if (i < row.size() - 1) {
                    outputFile << ",";
                }
            }

            // Ajout d'un saut de ligne après chaque ligne de la matrice
            outputFile << std::endl;
        }

        // Fermeture du fichier
        outputFile.close();

    }
    allValidationInfos.clear();}

std::vector<uint16_t> ArmLearnWrapper::getMotorsPos() {

    // Get device states
    auto deviceStates = DeviceLearner::getDeviceState();
    std::vector<uint16_t> motorPos;
    for (auto &deviceState : deviceStates) {
        for (unsigned short &value : deviceState) {
            // Save the motor position
            motorPos.emplace_back(value);
        }
    }
    return motorPos;
}

void ArmLearnWrapper::setgeneration(int newGeneration){
    generation = newGeneration;
}

std::vector<uint16_t> ArmLearnWrapper::getInitStartingPos(){
    return initStartingPos;
}

void ArmLearnWrapper::setInitStartingPos(std::vector<uint16_t> newInitStartingPos){
    initStartingPos = newInitStartingPos;
}

double ArmLearnWrapper::getCurrentMaxLimitTarget(){
    return currentMaxLimitTarget;
}

double ArmLearnWrapper::getCurrentMaxLimitStartingPos(){
    return currentMaxLimitStartingPos;
}


double ArmLearnWrapper::getDistance(){
    
    // Get the cartiesion coordonates of the arm
    std::vector<double> cartesianCoords = converter->computeServoToCoord(getMotorsPos())->getCoord();

    auto target = this->currentTarget->getInput();

    // Compute and return the Distance with the target
    return computeSquaredError(target, cartesianCoords);
}
