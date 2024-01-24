

#include "armlearnEvaluationResult.h"
#include <iostream>

double Learn::ArmlearnEvaluationResult::getSuccess() const{
    return this->success;
}

double Learn::ArmlearnEvaluationResult::getDistance() const{
    return this->distance;
}

std::vector<int> Learn::ArmlearnEvaluationResult::getTrajReached(){
    return this->trajReached;
}

Learn::EvaluationResult& Learn::ArmlearnEvaluationResult::operator+=(
    const EvaluationResult& other)
{
    // Type Check (Must be done in all override)
    // This test will succeed in child class.
    const std::type_info& thisType = typeid(*this);
    if (typeid(other) != thisType) {
        throw std::runtime_error("Type mismatch between EvaluationResults.");
    }

    auto otherConverted = (const Learn::ArmlearnEvaluationResult&)other;

    // If the added type is Learn::ArmlearnEvaluationResult
    if (thisType == typeid(Learn::ArmlearnEvaluationResult)) {

        // Weighted addition of results
        this->result = this->result * (double)this->nbEvaluation +
                       otherConverted.result * (double)otherConverted.nbEvaluation;
        this->result /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Weighted addition of success
        this->success = this->success * (double)this->nbEvaluation +
                       otherConverted.success * (double)otherConverted.nbEvaluation;
        this->success /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Weighted addition of distance
        this->distance = this->distance * (double)this->nbEvaluation +
                       otherConverted.distance * (double)otherConverted.nbEvaluation;
        this->distance /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Addition ot nbEvaluation
        this->nbEvaluation += otherConverted.nbEvaluation;
    }

    return *this;
}


