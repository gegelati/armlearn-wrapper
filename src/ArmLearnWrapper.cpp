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
        cartesianPos.setDataAt(typeid(double), i, newCartesianCoords[i]);
        cartesianDif.setDataAt(typeid(double), i, this->currentTarget->getInput()[i] - newCartesianCoords[i]);
    }
}

std::vector<std::reference_wrapper<const Data::DataHandler>> ArmLearnWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.emplace_back(cartesianPos);
    result.emplace_back(cartesianDif);
    result.emplace_back(motorPos);
    return result;
}

void ArmLearnWrapper::doAction(uint64_t actionID) {

    std::vector<double> out;
    double step = M_PI / 180 * params.sizeAction; // discrete rotations of 1°
    // -> move step size in training parameters

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
    auto scaledOutput = device->scalePosition(out, -M_PI, M_PI);

    // changes relative coordinates to absolute
    for (int i = 0; i < 4; i++) {
        double inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());

        // Substract by 2048 to get the out value indacted by the action
        scaledOutput[i] = (scaledOutput[i] - 2048) + inputI;
    }

    double inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
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

    int test_moving = 0;

    std::vector<double> out;
    for (float &action : actions) {
        out.push_back(params.sizeAction * action * M_PI / 180);

        // We condider that below 0.1, the motor is not moving
        if(abs(action) < 0.1){
            test_moving++;
        }
    }
    out.push_back(0.0);
    out.push_back(0.0);

    
    isMoving = (test_moving < 4);

    // Scale the positions
    auto scaledOutput = device->scalePosition(out, -M_PI, M_PI);


    // changes relative coordinates to absolute
    for (int i = 0; i < 4; i++) {
        double inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        scaledOutput[i] = (scaledOutput[i] - 2048) + inputI;
    }

    double inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
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
    if (score > params.thresholdUpgrade){
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

    if(params.reachingObjectives){
        if(score > params.thresholdUpgrade){
            terminal = true;
            return 1000;
        }
    } else if(nbActionsInThreshold == 10 || !isMoving){
        terminal = true;
        if(score > params.thresholdUpgrade){
            return 1000;
        }
    }

    // Return distance divided by the initCurrentMaxLimitTarget (this will push the arm to stay in the initCurrentMaxLimitTarget)
    return -err * params.coefRewardMultiplication;
    
}


void ArmLearnWrapper::reset(size_t seed, Learn::LearningMode mode, uint16_t iterationNumber, uint64_t generationNumber) {

    // Get the right trajectories' map
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>* trajectories;
    
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

        if (testFind != trajToDelete.end()){
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


    if(!validation && limit < 750 && params.progressiveModeMotor){
        // Create distribution
        std::normal_distribution<double> distribution(2048, limit);
        
        // Get random motor coordonates
        i = (int16_t) std::max(std::min(distribution(gen), 4096.0),1.0);
        j = (int16_t) std::max(std::min(distribution(gen), 3071.0),1025.0);
        k = (int16_t) std::max(std::min(distribution(gen), 3071.0),1025.0);
        l = (int16_t) std::max(std::min(distribution(gen), 3071.0),1025.0);
    }
    else{
        i = (int16_t) (rng.getUnsignedInt64(1, 4094));
        j = (int16_t) (rng.getUnsignedInt64(1025, 3071));
        k = (int16_t) (rng.getUnsignedInt64(1025, 3071));
        l = (int16_t) (rng.getUnsignedInt64(1025, 3071));
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

armlearn::Input<int16_t> *ArmLearnWrapper::randomGoal(std::vector<uint16_t> startingPos, bool validation){

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
    return new armlearn::Input<int16_t>(
    {
        (int16_t) (newCartesianCoords[0]), //X
        (int16_t) (newCartesianCoords[1]), //Y
        (int16_t) (newCartesianCoords[2])}); //Z
}


void ArmLearnWrapper::customTrajectory(armlearn::Input<int16_t> *newGoal, std::vector<uint16_t> startingPos, bool validation) {

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
        double input = (double) *(this->cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
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

            auto targetInput = new armlearn::Input<int16_t>({
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