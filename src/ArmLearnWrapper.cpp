#include "ArmLearnWrapper.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <cstdlib> 

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
    if (params.actionSpeed) result.emplace_back(dataMotorSpeed);
    return result;
}

void ArmLearnWrapper::doAction(uint64_t actionID) {

    std::vector<double> motorAction;
    double step  = params.sizeAction;

    // Get the action
    switch (actionID) {
        case 0:
            motorAction = {step, 0, 0, 0, 0, 0};
            break;
        case 1:
            motorAction = {0, step, 0, 0, 0, 0};
            break;
        case 2:
            motorAction = {0, 0, step, 0, 0, 0};
            break;
        case 3:
            motorAction = {0, 0, 0, step, 0, 0};
            break;
        case 4:
            motorAction = {-step, 0, 0, 0, 0, 0};
            break;
        case 5:
            motorAction = {0, -step, 0, 0, 0, 0};
            break;
        case 6:
            motorAction = {0, 0, -step, 0, 0, 0};
            break;
        case 7:
            motorAction = {0, 0, 0, -step, 0, 0};
            break;
        case 8:
            motorAction = {0, 0, 0, 0, 0, 0};
            if(gegelatiRunning && !params.actionSpeed){
                isMoving=false;
            }
            break;

            // Following cases only when the hand is trained
        case 9:
            motorAction = {0, 0, 0, 0, step, 0};
            break;
        case 10:
            motorAction = {0, 0, 0, 0, 0, step};
            break;
        case 11:
            motorAction = {0, 0, 0, 0, -step, 0};
            break;
        case 12:
            motorAction = {0, 0, 0, 0, 0, -step};
            break;
    }

    // Execute the action
    executeAction(motorAction);

}

void ArmLearnWrapper::doActionContinuous(std::vector<float> actions) {

    // Get the action
    std::vector<double> motorAction;
    for (float &action : actions) {
        motorAction.push_back(round(params.sizeAction * action));
    }
    motorAction.push_back(0.0);
    motorAction.push_back(0.0);

    // Execute the action
    executeAction(motorAction);
}

void ArmLearnWrapper::executeAction(std::vector<double> motorAction){

    if(params.actionSpeed){
        // Change the speed of the motors
        for(int i=0; i<6; i++){
            motorSpeed[i] += motorAction[i];
            if(i < 4){
                dataMotorSpeed.setDataAt(typeid(double), i, motorSpeed[i]);
            }
        }

        // The new speed is the action
        motorAction = motorSpeed;
    }

    int nbMotorMoving = 0;
    bool givePenaltyMoveUnavailable = false;

    // Scale the positions : this return a vector of int between 0 and 4096 corresponding to the step
    auto scaledOutput = device->scalePosition({0, 0, 0, 0, 0, 0}, -M_PI, M_PI);

    // The new position of the first motor is calculated before the three others because the possibilities of the motors are different
    double inputI = (double) *(motorPos.getDataAt(typeid(double), 0).getSharedPointer<const double>());

    if(params.canDo360){
        scaledOutput[0] = static_cast<uint16_t>(motorAction[0] + inputI) % 4096;

       // If the position aimed is possible, change the value
    }else if (motorAction[0] + inputI >= 2  && motorAction[0] + inputI <= 4094){
        scaledOutput[0] = motorAction[0] + inputI;
        if(motorAction[0] != 0){
            nbMotorMoving++;
        }
    } else {
        // Else do not change the value.
        // The arm is not moving
        scaledOutput[0] = inputI;

        // only active for gegelati because SAC is not deterministic
        if(gegelatiRunning){
            isMoving=false;
        }

        // Give a penalty if the algorithm as taken an unavailable action
        givePenaltyMoveUnavailable = true;

    }

    
    
    for (int i = 1; i < 4; i++) {
        inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        // If the position aimed is possible, change the value
        if(motorAction[i] + inputI >= 1025 && motorAction[i] + inputI <= 3071){
            scaledOutput[i] = motorAction[i]  + inputI;

        if(motorAction[i] != 0){
            nbMotorMoving++;
        }
        } else {
            // Else do not change the value.
            // The arm is not moving
            scaledOutput[i] = inputI;


            // only active for gegelati because SAC is not deterministic
            if(gegelatiRunning){
                isMoving=false;
            }

            // Give a penalty if the algorithm as taken an unavailable action (only one penalty even with multiple action)
            givePenaltyMoveUnavailable = true;
        }
    }

    if(params.realSimulation && motorCollision(scaledOutput)){

        // only active for gegelati because SAC is not deterministic
        if(gegelatiRunning){
            isMoving=false;
        }

        // Give a penalty if the algorithm as taken an unavailable action (only one penalty even with multiple action)
        givePenaltyMoveUnavailable = true;
        for (int i = 0; i < 4; i++) {
            inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
            scaledOutput[i] = inputI;
        }
    }

    // TODO update this when the hand will be trained
    inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
    scaledOutput[4] = (scaledOutput[4] - 511) + inputI;
    inputI = (double) *(motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>());
    scaledOutput[5] = (scaledOutput[5] - 256) + inputI;

    auto validOutput = device->toValidPosition(scaledOutput);
    device->setPosition(validOutput); // Update position
    device->waitFeedback();


    computeInput(); // to update  positions

    // Get the cartiesion coordonates of the arm
    std::vector<double> cartesianCoords = converter->computeServoToCoord(getMotorsPos())->getCoord();

    auto target = this->currentTarget->getInput();

    // Compute and return the Distance with the target
    distance = computeSquaredError(target, cartesianCoords);

    nbActionsDone++;
    reward = computeReward(givePenaltyMoveUnavailable, nbMotorMoving); // Computation of reward
    score += reward;

    if(gegelatiRunning){
        score = -1 * getDistance();

        double range = (isValidation) ? params.rangeTarget : currentRangeTarget;
        if(-1 * score < range){
            score = (nbMaxActions - nbActionsDone) * params.coefRewardMultiplication;
        }
    }

    if(gegelatiRunning){
        updateAndCheckCycles();
    }

    if(params.testing){
        saveMotorPos();
    }

}

void ArmLearnWrapper::updateAndCheckCycles(){
    auto motorPos = getMotorsPos();
    auto testFind = std::find(memoryMotorPos.begin(), memoryMotorPos.end(), motorPos);
    if (testFind != memoryMotorPos.end()){
        isCycling = true;
    } else{
        memoryMotorPos.push_back(motorPos);
    }
}

void ArmLearnWrapper::saveMotorPos(){
    // Push back the motor position
    allMotorPos.push_back(getMotorsPos());
    if(terminal || nbActionsDone == nbMaxActions){

        // If terminal or end of episode, add time (in ms), score, distance, success and number of actions
        vectorValidationInfos.push_back(static_cast<int32_t>(((std::chrono::duration<double>)(std::chrono::system_clock::now() - *checkpoint)).count()*1000));
        vectorValidationInfos.push_back(static_cast<int32_t>(getScore()));
        vectorValidationInfos.push_back(static_cast<int32_t>(distance));
        vectorValidationInfos.push_back(static_cast<int32_t>((distance < params.rangeTarget) ? 1: 0));
        vectorValidationInfos.push_back(static_cast<int32_t>(nbActionsDone));

        // Add each motor positions
        for(auto motor_value: allMotorPos){
            vectorValidationInfos.push_back(motor_value[0]);
            vectorValidationInfos.push_back(motor_value[1]);
            vectorValidationInfos.push_back(motor_value[2]);
            vectorValidationInfos.push_back(motor_value[3]);
        }
        // Add the vector containing the inforamtions to the vector containing all the informations
        allValidationInfos.push_back(vectorValidationInfos);
    }
}

double ArmLearnWrapper::computeReward(bool givePenaltyMoveUnavailable, int nbMotorMoving) {

    // Compute Distance with the target
    auto err = getDistance();

    double range = (isValidation) ? params.rangeTarget : currentRangeTarget;


    // Tempory reward to force to stop close to the objective
    if (err < range){
        // Incremente a counter
        nbActionsInThreshold++;

    // If not close to the objective
    } else{
        // reset counter
        nbActionsInThreshold=0;
    }


    // If the arm is not moving, set terminal to true
    if(!isMoving || isCycling){
        terminal = true;
    }

    if(params.reachingObjectives){
        if(err < range){
            terminal = true;
            return 10;
        }
    } else if(nbActionsInThreshold == 10 || !isMoving){
        terminal = true;
        if(err < range){
            return 10;
        }
    }

    // If the arm is not moving anymore or is cycling, the reward is multiplied by the number of action normally to come
    double penaltyStopTooSoon = 1;
    if((!isMoving || isCycling) && gegelatiRunning){
        penaltyStopTooSoon = nbMaxActions - nbActionsDone;
    }

    // If the arm has done an unavailable move, the algorithm get a penalty
    double penaltyMoveUnavailable = 0;
    if (givePenaltyMoveUnavailable && (!gegelatiRunning || params.isScoreResult)){
        penaltyMoveUnavailable = params.penaltyMoveUnavailable;
    }

    double penaltySpeed = 0;
    if(params.actionSpeed){
        for(auto speed: motorSpeed){
            penaltySpeed += abs(speed);
        }
        penaltySpeed *= params.penaltySpeed;
    } else if (nbMotorMoving > 1){
        penaltySpeed = (nbMotorMoving - 1) * params.penaltySpeed;
    }

    // Return distance divided by the initCurrentMaxLimitTarget (this will push the arm to stay in the initCurrentMaxLimitTarget)
    return (- err * params.coefRewardMultiplication - penaltyMoveUnavailable - penaltySpeed) * penaltyStopTooSoon;

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



    // Change the starting position
    this->currentStartingPos = trajectories->at(iterationNumber).first;

    device->setPosition(*currentStartingPos); // Reset position
    device->waitFeedback();

    // Change the target
    this->currentTarget = trajectories->at(iterationNumber).second;
    computeInput();

    // Init environnement parameters
    reward = 0.0;
    score = 0.0;
    nbActionsDone = 0;
    terminal = false;
    nbActionsInThreshold=0;
    isMoving = true;
    isCycling = false;
    motorSpeed = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    isValidation = (mode==Learn::LearningMode::VALIDATION);
    memoryMotorPos.clear();
    distance = 0.0;


    // If we are testing the arm, we save the current trajectory
    if(params.testing){
        allMotorPos.clear();
        vectorValidationInfos.clear();

        for(auto val: *trajectories->at(iterationNumber).first){
            vectorValidationInfos.push_back(val);
        }

        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[0]);
        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[1]);
        vectorValidationInfos.push_back(trajectories->at(iterationNumber).second->getInput()[2]);

        checkpoint = std::make_shared<std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>>(std::chrono::system_clock::now());
    }
}


void ArmLearnWrapper::addToScoreTrajectories(int index, double score){
    scoreTrajectories.push_back(std::make_pair(index, score));
}

void ArmLearnWrapper::clearPropTrainingTrajectories(){

    // Do not clear and just return if the vector is empty
    if (trainingTrajectories.size() == 0)
        return;

    // if the proportion is 1 do not delete anything
    if (params.propTrajectoriesReused >= 1){
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

    //Sort list traj with the best result in first
    std::sort(scoreTrajectories.begin(), scoreTrajectories.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    std::vector<int> trajToDelete;
    for(int i=0; i < nbDeletedTrajectories; i++){
        // Either get incremental or predifined index
        trajToDelete.push_back((params.controlTrajectoriesDeletion) ? scoreTrajectories[i].first: i);
    }
    
    // run the trajectories set backward to avoid size issues
    for(int i = trainingTrajectories.size() - 1; i >= 0; --i) {

        auto testFind = std::find(trajToDelete.begin(), trajToDelete.end(), i);

        // If i is in the trajectories to delete, delete it
        if (testFind != trajToDelete.end()){
            auto iterToDelete = trainingTrajectories.begin();
            std::advance(iterToDelete, i);

            if (params.doRandomStartingPosition) delete iterToDelete->first;
            delete iterToDelete->second;

            trainingTrajectories.erase(iterToDelete);
        }


    }
    // Clear the vector
    scoreTrajectories.clear();
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

        // Get a new starting pos, either random, either the init one depending on doRandomStartingPos
        auto newStartingPos = (params.doRandomStartingPosition) ? randomStartingPos(true) : &initStartingPos;

        // Get a new random Goal
        auto newTarget = randomGoal(*newStartingPos, true);

        // add the pair startingPos and target to the vector
        validationTrajectories.push_back(std::make_pair(newStartingPos, newTarget));
    }
}

std::vector<uint16_t> ArmLearnWrapper::randomMotorPos(std::vector<double> cartesianGoal, bool validation, bool isTarget){

    double minDistance = 10000;
    std::vector<uint16_t> keepMotorPos;

    uint16_t i, j, k, l;
    std::vector<uint16_t> newMotorPos, validMotorPos;

    std::vector<double> cartesianPos;
    double distance = 0;

    for(int index=0; index < 10000; index++){

        auto limit = (isTarget) ? currentMaxLimitTarget: currentMaxLimitStartingPos;


        if(!validation && params.progressiveModeMotor){

            int16_t valueNeeded = 2048 % (int)params.sizeAction;
            i = (uint16_t) (valueNeeded + (int)(rng.getInt32(std::max(2.0, 2048 - limit) - valueNeeded, std::min(4094.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
            j = (uint16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
            k = (uint16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);
            l = (uint16_t) (valueNeeded + (int)(rng.getInt32(std::max(1025.0, 2048 - limit) - valueNeeded, std::min(3071.0, 2048 + limit) - valueNeeded) / params.sizeAction) * params.sizeAction);

        }
        else{
            // The calcul insure that the value sampled are possible
            // For exemple with params.sizeAction = 5,
            // valueNeeded = 3, then the value is sample between 1022 and 3061. The division/multiplication allow to round around 5
            // Then we add 3 again to be sure that the coordonates are possible
            int16_t valueNeeded = 2048 % (int)params.sizeAction;
            i = (uint16_t) (valueNeeded + (int)(rng.getInt32(1 - valueNeeded, 4096 - valueNeeded) / params.sizeAction) * params.sizeAction);
            j = (uint16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
            k = (uint16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
            l = (uint16_t) (valueNeeded + (int)(rng.getInt32(1025 - valueNeeded, 3071 - valueNeeded) / params.sizeAction) * params.sizeAction);
        }

        // Create the vector of motor positions
        newMotorPos = {i,j,k,l,512,256};

        validMotorPos = device->toValidPosition(newMotorPos);

        cartesianPos = converter->computeServoToCoord(validMotorPos)->getCoord();

        distance = computeSquaredError(cartesianPos, cartesianGoal);
        if(distance < minDistance){
            keepMotorPos = validMotorPos;
            minDistance = distance;
        }
    }

    return keepMotorPos;
}

std::vector<uint16_t> *ArmLearnWrapper::randomStartingPos(bool validation){

    std::vector<uint16_t> motorPos;
    std::vector<double> newStartingPos, cartesianGoal;

    bool distanceIsNotGood = false;
    bool handNotGood = false;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    size_t index;

    // Do one time then only while the distance is above the distance between the new starting position and the initial one
    do {
        // Get random cartesian goal
        size_t index = rng.getUnsignedInt64(0, dataTarget.size());
        auto cartesianGoal = dataTarget[index];

        // Get a random motor positions
        motorPos = randomMotorPos(cartesianGoal, validation, false);

        // Compute the cartesian coordonates of those motor positions
        newStartingPos = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance the new starting position and the initial one
        distance = computeSquaredError(converter->computeServoToCoord(initStartingPos)->getCoord(), newStartingPos);

        // Distance is not good if it is above the current limit and bellow the current range target
        distanceIsNotGood = (distance > currentMaxLimitStartingPos);

        // Hand is not good if the target is bellow 0 on z axis
        handNotGood = (params.realSimulation && motorCollision(motorPos));

        if(handNotGood){
            // Delete index because it will never be correct
            auto it = dataTarget.begin();
            std::advance(it, index);
            dataTarget.erase(it);
        }

    } while ((!validation && distanceIsNotGood && !params.progressiveModeMotor) || handNotGood);

    // Delete used index
    auto it = dataTarget.begin();
    std::advance(it, index);
    dataTarget.erase(it);

    return new std::vector<uint16_t>(motorPos);

}

armlearn::Input<double> *ArmLearnWrapper::randomGoal(std::vector<uint16_t> startingPos, bool validation){

    std::vector<uint16_t> motorPos;
    std::vector<double> newCartesianCoords, cartesianGoal;

    bool distanceIsNotGood = false;
    bool handNotGood = false;

    // Init the distance at -1 to be sure that the while condition never return true during validation
    double distance = -1;

    size_t index = 0;

    // Do one time then only while the distance is above the distance to browse
    do {
        // Get random cartesian goal
        index = rng.getUnsignedInt64(0, dataTarget.size());
        cartesianGoal = dataTarget[index];

        // Get a random motor positions
        motorPos = randomMotorPos(cartesianGoal, validation, true);

        // Compute the cartesian coordonates of those motor positions
        newCartesianCoords = converter->computeServoToCoord(motorPos)->getCoord();

        // Compute the distance to browse
        distance = computeSquaredError(converter->computeServoToCoord(startingPos)->getCoord(), newCartesianCoords);

        // Distance is not good if it is above the current limit and bellow the current range target
        distanceIsNotGood = (distance > currentMaxLimitTarget || distance < currentRangeTarget);

        // Hand is not good if the target is bellow 0 on z axis
        handNotGood = (params.realSimulation && motorCollision(motorPos) );

        if(handNotGood){
            // Delete index because it will never be correct
            auto it = dataTarget.begin();
            std::advance(it, index);
            dataTarget.erase(it);
        }

    } while ((!validation && distanceIsNotGood && !params.progressiveModeMotor) || handNotGood);

    // Delete used index
    auto it = dataTarget.begin();
    std::advance(it, index);
    dataTarget.erase(it);

    // Create the input to return
    return new armlearn::Input<double>(
    {
        (double) (newCartesianCoords[0]), //X
        (double) (newCartesianCoords[1]), //Y
        (double) (newCartesianCoords[2])}); //Z
}

void ArmLearnWrapper::loadTargetCSV() {

    std::string slashToAdd = (std::filesystem::exists("/params/trainParams.json")) ? "/": "";

    std::ifstream file((slashToAdd + "params/AllTarget.csv").c_str());
    if (!file.is_open()) {
        std::cerr << "Error: unable to open file" << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> values;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            // Utiliser la fonction de conversion de chaîne en double
            std::istringstream iss(token);
            double result;
            iss >> result;
            values.push_back(result);
        }

        dataTarget.push_back(values);
    }

    file.close();

    std::mt19937 rngData(params.seed);
    std::shuffle(dataTarget.begin(), dataTarget.end(), rngData);
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

bool ArmLearnWrapper::updateCurrentLimits(double bestResult, int nbIterationsPerPolicyEvaluation){
    // If the best TPG is above the range for upgrade
    if (bestResult < currentRangeTarget){

        // Incremente the counter for upgrading the max current limit
        counterIterationUpgrade += 1;

        // If the counter reach the number of iterations to upgrade
        if(counterIterationUpgrade == params.nbIterationsUpgrade){

            // Upgrade the limit of tagets
            if (params.progressiveModeTargets){
                if(params.progressiveRangeTarget){
                    currentRangeTarget = std::max(currentRangeTarget * params.coefficientUpgradeMult, currentRangeTarget + params.coefficientUpgradeAdd);
                    currentRangeTarget = std::max(currentRangeTarget, params.rangeTarget);
                } else{
                    currentMaxLimitTarget = std::min(currentMaxLimitTarget * params.coefficientUpgradeMult, currentMaxLimitTarget + params.coefficientUpgradeAdd);
                    currentMaxLimitTarget = std::min(currentMaxLimitTarget, 750.0d);
                }

            }


            // Upgrade the limit of starting positions
            if (params.progressiveModeStartingPos){
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos * params.coefficientUpgradeMult, currentMaxLimitStartingPos + params.coefficientUpgradeAdd);
                currentMaxLimitStartingPos = std::min(currentMaxLimitStartingPos, 750.0d);
            }


            counterIterationUpgrade = 0;

            // Update the training validation trajectories
            updateTrainingValidationTrajectories(nbIterationsPerPolicyEvaluation);

            return true;
        }
    } else{
        // Reset the counter
        counterIterationUpgrade = 0;

    }
    return false;
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

    std::vector<std::vector<double>> allValues;

    if (inFile.is_open()) {

        std::string line;

        // Lecture de chaque ligne du fichier
        while (std::getline(inFile, line)) {
            std::vector<double> values;
            std::istringstream streamLine(line);
            double value;

            // Lecture des valeurs dans la ligne
            while (streamLine >> value) {
                values.push_back(value);
            }

            // Ajout du vecteur de valeurs de la ligne au vecteur principal
            allValues.push_back(values);
        }

        // Close the file
        inFile.close();

    } else {
        std::cerr << "Error while openning file for loading validation trajectories" << std::endl;
    }

    for(auto values: allValues){

        if(values.size()>0){
            std::vector<uint16_t>* startingPos = new std::vector<uint16_t>();
            for(int i = 0; i < 6; i++){
                startingPos->push_back((uint16_t)values[i]);
            }
            auto targetInput = new armlearn::Input<double>({values[6], values[7], values[8]});

            validationTrajectories.push_back(std::make_pair(startingPos, targetInput));
        }
    }
}

void ArmLearnWrapper::logTestingTrajectories(bool usingGegelati){

    // Nom du fichier CSV
    std::string fileName = (params.testPath + ((usingGegelati) ? "/outputGegelati.csv": "/outputSAC.csv")).c_str();

    // Ouverture du fichier en mode écriture
    std::ofstream outputFile(fileName);

    // Vérification si le fichier est correctement ouvert
    if (outputFile.is_open()) {
        outputFile<<"armPos0,"<<"armPos1,"<<"armPos2,"<<"armPos3,"<<"armPos4,"<<"armPos5,";
        outputFile<<"targetPos0,"<<"targetPos1,"<<"targetPos2,"<<"Duration(ms),"<<"Score,"<<"Distance,"<<"Success,"<<"NbActions,"<<"MotorPos"<<std::endl;
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

double ArmLearnWrapper::getCurrentRangeTarget(){
    return currentRangeTarget;
}

double ArmLearnWrapper::getDistance(){
    return distance;
}

bool ArmLearnWrapper::motorCollision(std::vector<uint16_t> newMotorPos){

    // Constant
    uint16_t length_base = 125;
    uint16_t length_shoulder = 142;
    uint16_t length_elbow = 142;
    uint16_t length_wrist = 155;

    uint16_t displacement = 49;

    double angle_base = (double) newMotorPos[0] / 4096 * 360;
    double angle_shoulder = ((double) newMotorPos[1] - 1024) / 2048 * 180;
    double angle_elbow = ((double) newMotorPos[2] - 1024) / 2048 * 180;
    double angle_wrist = ((double) newMotorPos[3] - 1024) / 2048 * 180;

    double radiant_angle_base = angle_base / 180 * M_PI;
    double radiant_angle_shoulder = angle_shoulder / 180 * M_PI;
    double radiant_angle_elbow = M_PI - angle_elbow / 180 * M_PI;
    double radiant_angle_wrist = M_PI/2 - angle_wrist / 180 * M_PI;

    double val1_side = std::cos(radiant_angle_shoulder) * length_shoulder;
    double val2_side = val1_side + std::cos(radiant_angle_shoulder + M_PI/2) * displacement;
    double val3_side = val2_side + std::cos(radiant_angle_shoulder + radiant_angle_elbow) * length_elbow;
    double val4_side = val3_side + std::cos(radiant_angle_shoulder + radiant_angle_elbow + radiant_angle_wrist) * length_wrist;

    double val1_z = std::sin(radiant_angle_shoulder) * length_shoulder + length_base;
    double val2_z = val1_z + std::sin(radiant_angle_shoulder + M_PI/2) * displacement;
    double val3_z = val2_z + std::sin(radiant_angle_shoulder + radiant_angle_elbow) * length_elbow;
    double val4_z = val3_z + std::sin(radiant_angle_shoulder + radiant_angle_elbow + radiant_angle_wrist) * length_wrist;

    
    if(val1_z < 0 || val2_z < 0 || val3_z < 0 || val4_z < 0){
        return true;
    }

    double angle = radiant_angle_shoulder + radiant_angle_elbow + radiant_angle_wrist;
    double coordx1 = val3_side + std::cos(angle) * 160 + std::cos(angle - M_PI/2) * 24;
    double coordx2 = val3_side + std::cos(angle) * 160 + std::cos(angle - M_PI/2) * -32;
    double coordx3 = val3_side + std::cos(angle - M_PI/2) * -32;

    double coordy1 = val3_z + std::sin(angle) * 160 + std::sin(angle - M_PI/2) * 24;
    double coordy2 = val3_z + std::sin(angle) * 160 + std::sin(angle - M_PI/2) * -32;
    double coordy3 = val3_z + std::sin(angle - M_PI/2) * -32;

    angle = radiant_angle_shoulder + radiant_angle_elbow;
    double coordx4 = val3_side + std::cos(angle) * 10 + std::cos(angle - M_PI/2) * 18;
    double coordx5 = val3_side + std::cos(angle) * 10 + std::cos(angle - M_PI/2) * -18 ;
    double coordx6 = val2_side + std::cos(angle - M_PI/2) * -18;
    
    double coordy4 = val3_z + std::sin(angle) * 10 + std::sin(angle - M_PI/2) * 18;
    double coordy5 = val3_z + std::sin(angle) * 10 + std::sin(angle - M_PI/2) * -18;
    double coordy6 = val2_z + std::sin(angle - M_PI/2) * -18;

    if(coordy1 < 0 || coordy2 < 0){
        return true;
    }

    std::vector<std::vector<double>> armSegments;
    armSegments.push_back({coordx1, coordx2, coordy1, coordy2});
    armSegments.push_back({coordx3, coordx2, coordy3, coordy2});
    armSegments.push_back({coordx4, coordx5, coordy4, coordy5});
    armSegments.push_back({coordx6, coordx5, coordy6, coordy5});

    for(auto armSeg: armSegments){
        for(auto val: armSeg){
        }

        for(auto baseSeg: baseSegments){
            if(hasCollision(armSeg, baseSeg)){
                return true;
            }
        }
    }
    return false;

}

bool ArmLearnWrapper::hasCollision(std::vector<double> armSegment, std::vector<double> baseSegment){

    // Vector contain xA, xB, yA, yB
    if(baseSegment[2] == baseSegment[3]){

        if (std::min(armSegment[0], armSegment[1]) > std::max(baseSegment[0], baseSegment[1])){
            return false;
        }
        if (std::max(armSegment[0], armSegment[1]) < std::min(baseSegment[0], baseSegment[1])){
            return false;
        }

        if (std::max(armSegment[2], armSegment[3]) < baseSegment[2]){
            return false;
        }
        if (std::min(armSegment[2], armSegment[3]) > baseSegment[2]){
            return false;
        }


        if(armSegment[0] == armSegment[1]){
            return true;
        }

        //Calcul equation y = ax + b
        double a = (armSegment[3] - armSegment[2]) / (armSegment[1] - armSegment[0]);
        double b = armSegment[2] - a * armSegment[0];

        double val_x = (baseSegment[2] - b)/a;
        if (val_x > std::max(baseSegment[0], baseSegment[1])){
            return false;
        }
        if (val_x < std::min(baseSegment[0], baseSegment[1])){
            return false;
        }


    }
    else if(baseSegment[0] == baseSegment[1]){

        if (std::min(armSegment[2], armSegment[3]) > std::max(baseSegment[2], baseSegment[3])){
            return false;
        }
        if (std::max(armSegment[2], armSegment[3]) < std::min(baseSegment[2], baseSegment[3])){
            return false;
        }

        if (std::max(armSegment[0], armSegment[1]) < baseSegment[0]){
            return false;
        }
        if (std::min(armSegment[0], armSegment[1]) > baseSegment[0]){
            return false;
        }

        if(armSegment[0] == armSegment[1]){
            return (armSegment[1] == baseSegment[1]);
        }

        //Calcul equation y = ax + b
        double a = (armSegment[3] - armSegment[2]) / (armSegment[1] - armSegment[0]);
        double b = armSegment[2] - a * armSegment[0];

        double val_y = a * baseSegment[0] + b;
        if (val_y > std::max(baseSegment[2], baseSegment[3])){
            return false;
        }
        if (val_y < std::min(baseSegment[2], baseSegment[3])){
            return false;
        }
    }



    return true;


}