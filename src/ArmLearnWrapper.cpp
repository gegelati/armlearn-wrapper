#include "ArmLearnWrapper.h"
#include <iostream>
#include <fstream>
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
    double step = M_PI / 180; // discrete rotations of 1°
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

}

void ArmLearnWrapper::doActionContinuous(std::vector<float> actions) {

    std::vector<double> out;
    for (float &action : actions) {
        out.push_back(1.0 * action * M_PI / 180);
    }
    out.push_back(0.0);
    out.push_back(0.0);



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
}

double ArmLearnWrapper::computeReward() {

    // Get the cartiesion coordonates of the arm
    std::vector<double> cartesianCoords;
    for (int i = 0; i < cartesianPos.getLargestAddressSpace(); i++) {
        cartesianCoords.emplace_back(
                (double) *cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    auto target = this->currentTarget->getInput();

    // Compute que Distance with the target
    auto err = computeSquaredError(target, cartesianCoords);

    /// Compute the number of actions taken in the episode divide by the maximum number of actions takeable in an episode
    /// This ratio is multiplied by a coefficient that allow to choose the impact of this ratio on the reward
    double valNbIterations = params.coefRewardNbIterations * (static_cast<double>(nbActions) / nbMaxActions);

    // set Score
    score = -1 * err;

    // Tempory reward to force to stop close to the objective
    if (score > params.thresholdUpgrade){

        // Incremente a counter
        nbActionsInThreshold++;

        // If the counter reach 10 or terminal is true (because the arm can stop and set terminal=true with action 8)
        if(nbActionsInThreshold == 10){
            terminal = true;
            return 1000;
        }
        // Else return 0 (still better than any reward not close to the objective)
        return -err/(currentMaxLimitTarget*10);

    // If not close to the objective
    } else{
        // reset counter
        nbActionsInThreshold=0;

        // Return distance divided by the initCurrentMaxLimitTarget (this will push the arm to stay in the currentMaxLimitTarget)
        return  -err/currentMaxLimitTarget;
    }
    
}


void ArmLearnWrapper::reset(size_t seed, Learn::LearningMode mode) {

    // Get the right iterator and trajectories' map
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>::iterator iterator;
    std::vector<std::pair<std::vector<uint16_t>*, armlearn::Input<int16_t>*>>* trajectories;
    
    switch (mode) {
        case Learn::LearningMode::TRAINING:
            iterator = trainingIterator;
            trajectories = &trainingTrajectories;
            break;
        case Learn::LearningMode::VALIDATION:
            iterator = validationIterator;
            trajectories = &validationTrajectories;
            break;
        case Learn::LearningMode::TESTING:
            iterator = trainingValidationIterator;
            trajectories = &trainingValidationTrajectories;
            break;
    }

    // Change the starting position
    this->currentStartingPos = iterator->first;

    device->setPosition(*currentStartingPos); // Reset position
    device->waitFeedback();

    // Change the target
    this->currentTarget = iterator->second;
    computeInput();

    // Incremente the iterator
    ++iterator;
    // If iterator is at the end, reset it
    if (iterator == trajectories->end()) iterator = trajectories->begin();

    // Init environnement parameters
    score = 0;
    nbActions = 0;
    terminal = false;
    nbActionsInThreshold=0;
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



    // Clear a define prortion of the training targets by giving the proportion of targets reused
    clearPropTrainingTrajectories();

    for (int i=0; i<nbTrajectories; i++){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (params.doRandomStartingPosition) ? randomStartingPos(false) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, false);

        // add the pair startingPos and target to the vector
        trainingTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }

    // Initiate the iterator of the trainingTrajectories
    trainingIterator = trainingTrajectories.begin();
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

    // Initiate the iterator of the trainingValidationTrajectories
    trainingValidationIterator = trainingValidationTrajectories.begin();
}

void ArmLearnWrapper::updateValidationTrajectories(int nbTrajectories){

    // Clear all the current validation trajectories
    std::for_each(validationTrajectories.begin(), validationTrajectories.end(), [this](auto& pair){
         if (this->params.doRandomStartingPosition) delete pair.first; // check doublon pointeur
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

    // Initiate the iterator of the validationTrajectories
    validationIterator = validationTrajectories.begin();
}

std::vector<uint16_t> ArmLearnWrapper::randomMotorPos(){
    uint16_t i, j, k, l;
    std::vector<uint16_t> newMotorPos, validMotorPos;

    // Get random motor coordonates
    i = (int16_t) (rng.getUnsignedInt64(1, 4094));
    j = (int16_t) (rng.getUnsignedInt64(1025, 3071));
    k = (int16_t) (rng.getUnsignedInt64(1025, 3071));
    l = (int16_t) (rng.getUnsignedInt64(1025, 3071));

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
        motorPos = randomMotorPos();

        // Compute the cartesian coordonates of those motor positions
        newStartingPos = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance the new starting position and the initial one
        distance = computeSquaredError(converter->computeServoToCoord(initStartingPos)->getCoord(), newStartingPos);

    } while (!validation && distance > currentMaxLimitStartingPos);

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
        motorPos = randomMotorPos();

        // Compute the cartesian coordonates of those motor positions
        newCartesianCoords = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance to browse
        distance = computeSquaredError(converter->computeServoToCoord(startingPos)->getCoord(), newCartesianCoords);

    } while (!validation && distance > currentMaxLimitTarget);
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
    if (bestResult > params.thresholdUpgrade){

        // Incremente the counter for upgrading the max current limit
        counterIterationUpgrade += 1;

        // If the counter reach the number of iterations to upgrade
        if(counterIterationUpgrade == params.nbIterationsUpgrade){

            // Upgrade the limit of tagets
            if (params.progressiveModeTargets){
                currentMaxLimitTarget = std::min(currentMaxLimitTarget * params.coefficientUpgrade, currentMaxLimitTarget + 30);
                currentMaxLimitTarget = std::min(currentMaxLimitTarget, 1000.0d);
            }


            // Upgrade the limit of starting positions
            if (params.progressiveModeStartingPos){
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos * params.coefficientUpgrade, currentMaxLimitStartingPos + 30);
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos, 200.0d);
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
    toLog << trainingIterator->second->getInput()[0] << " ; ";
    toLog << trainingIterator->second->getInput()[1] << " ; ";
    toLog << trainingIterator->second->getInput()[2] << " ; ";
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
    res << trainingIterator->second->getInput()[0] << " ; ";
    res << trainingIterator->second->getInput()[1] << " ; ";
    res << trainingIterator->second->getInput()[2] << " ; ";
    res << ")";

    return res.str();
}

Learn::LearningEnvironment *ArmLearnWrapper::clone() const {
    return new ArmLearnWrapper(*this);
}

void ArmLearnWrapper::saveValidationTrajectories() {
    // Create file
    std::ofstream outFile("ValidationTrajectories.txt");

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
        if (this->params.doRandomStartingPosition) delete pair.first; // check doublon pointeur
        delete pair.second;
    }); 
    validationTrajectories.clear();

    // Get file
    std::ifstream inFile("ValidationTrajectories.txt");

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

        // Initiate the iterator of the validationTrajectories
        validationIterator = validationTrajectories.begin();
    } else {
        std::cerr << "Error while openning file for loading validation trajectories" << std::endl;
    }
}

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
