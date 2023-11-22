#include "ArmLearnWrapper.h"
#include <iostream>

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

    // Get the cartesian coordonates of the motos
    auto newCartesianCoords = converter->computeServoToCoord(newMotorPos)->getCoord();

    // For each motor, save the position and the relative position with the target
    for (int i = 0; i < newCartesianCoords.size(); i++) {
        cartesianPos.setDataAt(typeid(double), i, newCartesianCoords[i]);
        cartesianDif.setDataAt(typeid(double), i, this->currentTarget->getInput()[i] - newCartesianCoords[i]);
    }
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
            // Since the arm is not moving, its position will remain identical, and
            // the action will keep being selected. So, terminate the eval.
            this->terminal = true;
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

    // Scale the positions
    auto scaledOutput = device->scalePosition(out, -M_PI, M_PI);

    // changes relative coordinates to absolute
    for (int i = 0; i < 4; i++) {
        double inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        scaledOutput[i] = (scaledOutput[i] - 2048) + inputI;
    }
    // TODO find out scaledOutput[0] += 1; // kdesnos: I don't know what the purpose of this line is

    double inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
    scaledOutput[4] = (scaledOutput[4] - 511) + inputI;
    inputI = (double) *(motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>());
    scaledOutput[5] = (scaledOutput[5] - 256) + inputI;

    auto validOutput = device->toValidPosition(scaledOutput);
    device->setPosition(validOutput); // Update position
    device->waitFeedback();

    computeInput(); // to update  positions

    nbActions++;
    auto reward = computeReward(); // Computation of reward

    score = reward; //Change reward
}

double ArmLearnWrapper::computeReward() {

    // Get the cartiesion coordonates of the arm
    std::vector<double> cartesianCoords;
    for (int i = 0; i < cartesianPos.getLargestAddressSpace(); i++) {
        cartesianCoords.emplace_back(
                (double) *cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    auto target = this->currentTarget->getInput();

    // Calcul que Distance with the target
    auto err = computeSquaredError(target, cartesianCoords);

    /// Calcul the number of actions taken in the episode divide by the maximum number of actions takeable in an episode
    /// This ratio is multiplied by a coefficient that allow to choose the impact of this ratio on the reward
    double valNbIterations = coefRewardNbIterations * (static_cast<double>(nbActions) / nbMaxActions);
    

    return -1 * (err + valNbIterations);
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
}

std::vector<std::reference_wrapper<const Data::DataHandler>> ArmLearnWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.emplace_back(cartesianDif);
    result.emplace_back(motorPos);
    return result;
}

void ArmLearnWrapper::clearPropTrainingTrajectories(double prop, bool doRandomStartingPos){

    // Do not clear and just return if the vector is empty
    if (trainingTrajectories.size() == 0)
        return;

    // Clear all and return if the proportion is 1 (or above, even if it should not be higher than 1)
    if (prop >= 1){
        trainingTrajectories.clear();
        return;
    }

    // Calcul the number of trajectories we want to delete
    auto nbDeletedTrajectories = static_cast<int>(round(trainingTrajectories.size() * (1-prop)));

    // Take an iterator to reach the last deleted trajectory
    auto it = trainingTrajectories.begin();
    std::advance(it, nbDeletedTrajectories);

    // Delete all the pointers from memory
    std::for_each(trainingTrajectories.begin(), it, [doRandomStartingPos](auto& pair){
         if (doRandomStartingPos) delete pair.first; // check doublon pointeur
         delete pair.second;
    }); 
    // Delete then the pair in the vector
    trainingTrajectories.erase(trainingTrajectories.begin(), it);
}


double ArmLearnWrapper::getScore() const {
    return score;
}


bool ArmLearnWrapper::isTerminal() const {
    return terminal;
}

bool ArmLearnWrapper::isCopyable() const {
    return true;
}


void ArmLearnWrapper::updateTrainingTrajectories(int nbTrajectories, bool doRandomStartingPos, double propTrajectoriesReused,
                                                 double limitTargets, double limitStartingPos){



    // Clear a define prortion of the training targets by giving the proportion of targets reused
    clearPropTrainingTrajectories(propTrajectoriesReused, doRandomStartingPos);

    while(trainingTrajectories.size() < nbTrajectories){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (doRandomStartingPos) ? randomStartingPos(false, limitStartingPos) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, false, limitTargets);

        // add the pair startingPos and target to the vector
        trainingTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }

    // Initiate the iterator of the trainingTrajectories
    trainingIterator = trainingTrajectories.begin();
}

void ArmLearnWrapper::updateTrainingValidationTrajectories(int nbTrajectories, bool doRandomStartingPos, double propTrajectoriesReused,
                                                           double limitTargets, double limitStartingPos){


    // Clear a define prortion of the training targets by giving the proportion of targets reused
    trainingValidationTrajectories.clear();

    while(trainingValidationTrajectories.size() < nbTrajectories){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (doRandomStartingPos) ? randomStartingPos(false, limitStartingPos) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, false, limitTargets);

        // add the pair startingPos and target to the vector
        trainingValidationTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }

    // Initiate the iterator of the trainingValidationTrajectories
    trainingValidationIterator = trainingValidationTrajectories.begin();
}

void ArmLearnWrapper::updateValidationTrajectories(int nbTrajectories, bool doRandomStartingPos){

    // Clear all the current validation trajectories
    validationTrajectories.clear();

    while(validationTrajectories.size() < nbTrajectories){

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
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

std::vector<uint16_t> *ArmLearnWrapper::randomStartingPos(bool validation, double maxLength){

    std::vector<uint16_t> motorPos;
    std::vector<double> newStartingPos;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    // Do one time then only while the distance is above the distance between the new starting position and the initial one
    do {
        // Get a random motor positions
        motorPos = randomMotorPos();

        // Calculate the cartesian coordonates of those motor positions
        newStartingPos = converter->computeServoToCoord(motorPos)->getCoord();

        if(!validation)
            // Calculate the distance the new starting position and the initial one
            distance = computeSquaredError(converter->computeServoToCoord(initStartingPos)->getCoord(), newStartingPos);

    } while (distance > maxLength);

    return new std::vector<uint16_t>(motorPos);

}

armlearn::Input<int16_t> *ArmLearnWrapper::randomGoal(std::vector<uint16_t> startingPos, bool validation, double maxLength){

    std::vector<uint16_t> motorPos;
    std::vector<double> newCartesianCoords;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    // Do one time then only while the distance is above the distance to browse
    do {
        // Get a random motor positions
        motorPos = randomMotorPos();

        // Calculate the cartesian coordonates of those motor positions
        newCartesianCoords = converter->computeServoToCoord(motorPos)->getCoord();

        if(!validation)
            // Calculate the distance to browse
            distance = computeSquaredError(converter->computeServoToCoord(startingPos)->getCoord(), newCartesianCoords);

    } while (distance > maxLength);

    // Create the input to return
    return new armlearn::Input<int16_t>(
    {
        (int16_t) (newCartesianCoords[0]), //X
        (int16_t) (newCartesianCoords[1]), //Y
        (int16_t) (newCartesianCoords[2])}); //Z
}


void ArmLearnWrapper::customTrajectory(armlearn::Input<int16_t> *newGoal, std::vector<uint16_t> startingPos, bool validation) {

    // Get the right vectpr of trajectories
    auto trajectories = (validation) ? trainingTrajectories : validationTrajectories;

    // Delete the first key/value pair if the vector is not empty
    if(trajectories.size() > 0){
        auto iterator = trajectories.begin();
        delete iterator->first;
        delete iterator->second;
        trajectories.erase(iterator);
    }
    
    // Add the custom target with the corresponding starting position
    trajectories.push_back(std::make_pair(&startingPos, newGoal));
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

