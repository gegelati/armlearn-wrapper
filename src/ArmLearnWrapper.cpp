#include "ArmLearnWrapper.h"

void ArmLearnWrapper::computeInput() {
    auto deviceStates = DeviceLearner::getDeviceState();
    std::vector<uint16_t> newMotorPos;
    int indInput = 0;
    for (auto &deviceState : deviceStates) {
        for (unsigned short &value : deviceState) {
            motorPos.setDataAt(typeid(double), indInput, value);
            newMotorPos.emplace_back(value);
            indInput++;
        }
    }

    auto newCartesianCoords = converter->computeServoToCoord(newMotorPos)->getCoord();

    for (int i = 0; i < newCartesianCoords.size(); i++) {
        cartesianPos.setDataAt(typeid(double), i, newCartesianCoords[i]);
        cartesianDif.setDataAt(typeid(double), i, this->currentTarget->getInput()[i] - newCartesianCoords[i]);
    }
}

void ArmLearnWrapper::doAction(uint64_t actionID) {
    std::vector<double> out;
//    double step = M_PI / 180; // discrete rotations of some °
//    double step = M_PI / 360;
    double step = M_PI / 720;

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


    auto scaledOutput = device->scalePosition(out, -M_PI, M_PI);

    // changes relative coordinates to absolute
    for (int i = 0; i < 4; i++) {
        double inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        scaledOutput[i] = (scaledOutput[i] - 2048) + inputI;
    }
    scaledOutput[0] += 1; // kdesnos: I don't know what the purpose of this line is

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



    score = reward;
}

double ArmLearnWrapper::computeReward() {
    std::vector<uint16_t> motorCoords;
    for (int i = 0; i < motorPos.getLargestAddressSpace(); i++) {
        motorCoords.emplace_back((uint16_t) *motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    std::vector<double> cartesianCoords;
    for (int i = 0; i < cartesianPos.getLargestAddressSpace(); i++) {
        cartesianCoords.emplace_back(
                (double) *cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    auto target = this->currentTarget->getInput();

    if (!device->validPosition(motorCoords)) return VALID_COEFF;
    auto err = computeSquaredError(target, cartesianCoords);
    //if (abs(err) < 1) {
    //    terminal = true;
    //}
/*
    if(err<5 && nbActions==999)
    std::cout<<toString()<<std::endl;*/

//    if(learningtarget == Learn::LearningMode::VALIDATION){
//        std::ofstream PointCloud;
//        PointCloud.open("PointCloud.ods",std::ios::app);
//
//        PointCloud << cartesianCoords[0] << "\t" << cartesianCoords[1] << "\t" << cartesianCoords[2] << "\t";
//        PointCloud << target[0] << "\t" << target[1] << "\t" << target[2] << "\t" << -1*err << "\t" << std::endl;
//    }

//    if(learningtarget == Learn::LearningMode::VALIDATION){
//        std::ofstream PointCloudXt;
//        PointCloudXt.open("PointCloudXt.m",std::ios::app);
//        PointCloudXt << target[0] << "  " ;
//
//        std::ofstream PointCloudYt;
//        PointCloudYt.open("PointCloudYt.m",std::ios::app);
//        PointCloudYt << target[1] << "  " ;
//
//        std::ofstream PointCloudXa;
//        PointCloudXa.open("PointCloudXa.m",std::ios::app);
//        PointCloudXa << cartesianCoords[0] << "  " ;
//
//        std::ofstream PointCloudYa;
//        PointCloudYa.open("PointCloudYa.m",std::ios::app);
//        PointCloudYa << cartesianCoords[1] << "  " ;
//
//        std::ofstream PointCloudE;
//        PointCloudE.open("PointCloudE.m",std::ios::app);
//        PointCloudE << -1*err << "  " ;
//
//
//    }

    return -1 * err;
}


void ArmLearnWrapper::reset(size_t seed, Learn::LearningMode mode) {
    device->setPosition(startingPos); // Reset position
    device->waitFeedback();

    learningtarget = mode;
    swapGoal(mode);

    computeInput();

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

double ArmLearnWrapper::getScore() const {
    if(learningtarget == Learn::LearningMode::VALIDATION){
    std::vector<uint16_t> motorCoords;
    for (int i = 0; i < motorPos.getLargestAddressSpace(); i++) {
        motorCoords.emplace_back((uint16_t) *motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    std::vector<double> cartesianCoords;
    for (int i = 0; i < cartesianPos.getLargestAddressSpace(); i++) {
        cartesianCoords.emplace_back(
                (double) *cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
    }
    auto target = this->currentTarget->getInput();


        std::ofstream PointCloud;
        PointCloud.open("PointCloud.csv",std::ios::app);

        PointCloud << cartesianCoords[0] << ";" << cartesianCoords[1] << ";" << cartesianCoords[2] << ";";
        PointCloud << target[0] << ";" << target[1] << ";" << target[2] << ";" << score << std::endl;
    }

    return score;
}


bool ArmLearnWrapper::isTerminal() const {
    return terminal;
}

bool ArmLearnWrapper::isCopyable() const {
    return true;
}

void ArmLearnWrapper::swapGoal(Learn::LearningMode mode) {
    auto &targets = (mode == Learn::LearningMode::TRAINING) ? trainingTargets : validationTargets;
    if (targets.size() > 1) {
        std::rotate(targets.begin(), targets.begin() + 1, targets.end());
    }

    learningtarget = mode;
    this->currentTarget = targets.at(0);
}

armlearn::Input<int16_t> *ArmLearnWrapper::randomGoal(std::vector<std::string> tpara) {
    int Xa = 0,Xb = 0,Ya = 346,Yb = 346,Za = 267,Zb = 267;

    if(tpara[1]=="close"){
        Xa = Xa-50;
        Xb = Xb+50;

        Ya = Ya-50;
        Yb = Yb+50;

        Za = Za-50;
        Zb = Zb+50;
    }
    if(tpara[1]=="large"){
        Xa = -100;
        Xb = 100;

        Ya = 100;
        Yb = 350;

        Za = Za-150;
        Zb = Zb+150;
    }

    if(tpara[1]=="full"){
        Xa = -500;
        Xb = 500;

        Ya = -500;
        Yb = 500;

        Za = -500;
        Zb = 500;
    }

    if(tpara[0]=="2D"){
        Za = 267;
        Zb = 267;
    }

//            (int16_t) (rng.getUnsignedInt64(-50, 50)), //X
//            (int16_t) (rng.getUnsignedInt64(276, 366)), //Y
//            (int16_t) (rng.getUnsignedInt64(197, 197))}); //Z

    //{(int16_t) (rng.getUnsignedInt64(-200, 200)), (int16_t) (rng.getUnsignedInt64(-200, 200)),
    //(int16_t) (rng.getUnsignedInt64(-150, 400))});
    return new armlearn::Input<int16_t>(
            {
                    (int16_t) (rng.getUnsignedInt64(Xa, Xb)), //X
                    (int16_t) (rng.getUnsignedInt64(Ya, Yb)), //Y
                    (int16_t) (rng.getUnsignedInt64(Za, Zb))}); //Z
}

void ArmLearnWrapper::customGoal(armlearn::Input<int16_t> *newGoal) {
    delete trainingTargets.at(0);
    trainingTargets.erase(trainingTargets.begin());
    trainingTargets.emplace(trainingTargets.begin(), newGoal);
}

std::string ArmLearnWrapper::newGoalToString() const {
    std::stringstream toLog;
    toLog << " - (new goal : ";
    toLog << trainingTargets[0]->getInput()[0] << " ; ";
    toLog << trainingTargets[0]->getInput()[1] << " ; ";
    toLog << trainingTargets[0]->getInput()[2] << " ; ";
    toLog << ")" << std::endl;
    return toLog.str();
}

std::string ArmLearnWrapper::toString() const {
    std::stringstream res;
    for (int i = 0; i < 6; i++) {
        double input = (double) *(this->motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        res << input << " ; ";
    }

    res << "    -->    ";
    for (int i = 0; i < 3; i++) {
        double input = (double) *(this->cartesianPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        res << input << " ; ";
    }
    res << " - (goal : ";
    res << trainingTargets[0]->getInput()[0] << " ; ";
    res << trainingTargets[0]->getInput()[1] << " ; ";
    res << trainingTargets[0]->getInput()[2] << " ; ";
    res << ")";

    return res.str();
}

Learn::LearningEnvironment *ArmLearnWrapper::clone() const {
    return new ArmLearnWrapper(*this);
}

std::vector<uint16_t> ArmLearnWrapper::getMotorsPos() {
    auto deviceStates = DeviceLearner::getDeviceState();
    std::vector<uint16_t> motorPos;
    for (auto &deviceState : deviceStates) {
        for (unsigned short &value : deviceState) {
            motorPos.emplace_back(value);
        }
    }
    return motorPos;
}

void ArmLearnWrapper::changeStartingPos(std::vector<uint16_t> newStartingPos) {
    startingPos = newStartingPos;
}
