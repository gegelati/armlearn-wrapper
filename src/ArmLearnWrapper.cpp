#include "ArmLearnWrapper.h"

void ArmLearnWrapper::computeInput() {
    auto deviceStates = DeviceLearner::getDeviceState();
    int indInput = 0;
    for (auto &deviceState : deviceStates) {
        for (unsigned short &value : deviceState) {
            motorPos.setDataAt(typeid(double), indInput, value);
            indInput++;
        }
    }
}

void ArmLearnWrapper::doAction(uint64_t actionID) {
    std::vector<double> out;
    double step = M_PI / 60; // discrete rotations of some °

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
            out = {0, 0, 0, 0, step, 0};
            break;
        case 5:
            out = {0, 0, 0, 0, 0, step};
            break;
        case 6:
            out = {-step, 0, 0, 0, 0, 0};
            break;
        case 7:
            out = {0, -step, 0, 0, 0, 0};
            break;
        case 8:
            out = {0, 0, -step, 0, 0, 0};
            break;
        case 9:
            out = {0, 0, 0, -step, 0, 0};
            break;
        case 10:
            out = {0, 0, 0, 0, -step, 0};
            break;
        case 11:
            out = {0, 0, 0, 0, 0, -step};
            break;
        case 12:
            out = {0, 0, 0, 0, 0, 0};
            break;
    }


    auto scaledOutput = device->scalePosition(out, -M_PI, M_PI);

    // changes relative coordinates to absolute
    for (int i = 0; i < 4; i++) {
        double inputI = (double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        scaledOutput[i] = (scaledOutput[i] - 2048) + inputI;
    }
    scaledOutput[0] += 1;
    double inputI = (double) *(motorPos.getDataAt(typeid(double), 4).getSharedPointer<const double>());
    scaledOutput[4] = (scaledOutput[4] - 511) + inputI;
    inputI = (double) *(motorPos.getDataAt(typeid(double), 5).getSharedPointer<const double>());
    scaledOutput[5] = (scaledOutput[5] - 256) + inputI;

    auto validOutput = device->toValidPosition(scaledOutput);

    std::vector<uint16_t> positionOutput;
    for (auto ptr = validOutput.cbegin(); ptr < validOutput.cend(); ptr++) positionOutput.push_back(*ptr);

    auto newCartesianCoords = converter->computeServoToCoord(positionOutput)->getCoord();

    auto reward = computeReward(targets[0]->getInput(), newCartesianCoords); // Computation of reward

    device->setPosition(validOutput); // Update position
    device->waitFeedback();
    //pyLearn((*ptr)->getInput(), (*ptr)->getOutput(), (*ptr)->getReward(), nextInput, std::pow(DECREASING_REWARD, i)); // Decrease value of reward as the state is closer to initial state
    score += reward;

    for(int i=0; i<newCartesianCoords.size();i++){}
    computeInput();
}

template<class R, class T>
double ArmLearnWrapper::computeReward(const std::vector<R> &target, const std::vector<T> &pos) const {
    std::vector<T> positionOutput;
    for (auto ptr = pos.cbegin(); ptr < pos.cend(); ptr++) positionOutput.push_back(*ptr);

    if (!device->validPosition(positionOutput)) return VALID_COEFF;

    auto newCoords = converter->computeServoToCoord(positionOutput)->getCoord();

    auto err = computeSquaredError(target, newCoords);
    // if(abs(err) < LEARN_ERROR_MARGIN) return -VALID_COEFF;
    return -TARGET_COEFF * err - MOVEMENT_COEFF * computeSquaredError(device->getPosition(), pos);
}


void ArmLearnWrapper::reset(size_t seed, Learn::LearningMode mode) {
    device->goToBackhoe(); // Reset position
    device->waitFeedback();
    computeInput();
    score = 0;
}

std::vector<std::reference_wrapper<const Data::DataHandler>> ArmLearnWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.emplace_back(motorPos);
    return result;
}

double ArmLearnWrapper::getScore() const {
    /* if(score>-167.56){
         std::cout<<"coords : "<<std::endl;
         for(int i=0; i<6; i++){
             std::cout<<(double) *(motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>())<<" ; ";
         }
         std::cout<<std::endl;
     }*/
    return score;
}


bool ArmLearnWrapper::isTerminal() const {
    return false;
}

bool ArmLearnWrapper::isCopyable() const {
    return true;
}

std::string ArmLearnWrapper::toString() const {
    std::stringstream res;
    std::vector<uint16_t> in(6);
    for (int i = 0; i < 6; i++) {
        double input = (double) *(this->motorPos.getDataAt(typeid(double), i).getSharedPointer<const double>());
        res << input << " ; ";
        in[i] = input;
    }

    res << "    -->    ";
    auto newCoords = converter->computeServoToCoord(in)->getCoord();
    for (auto i : newCoords) {
        res << (int)i << " ; ";
    }

    return res.str();
}

Learn::LearningEnvironment *ArmLearnWrapper::clone() const {
    return new ArmLearnWrapper(*this);
}
