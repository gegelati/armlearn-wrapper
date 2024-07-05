

#include "armlearnEvaluationResult.h"
#include <iostream>

double Learn::ArmlearnEvaluationResult::getSuccess() const{
    return this->success;
}

double Learn::ArmlearnEvaluationResult::getDistance() const{
    return this->distance;
}

double Learn::ArmlearnEvaluationResult::getNbActivatedTeam() const{
    return this->nbActivatedTeam;
}

double Learn::ArmlearnEvaluationResult::getNbActivatedTeamRatio() const{
    return this->nbActivatedTeamRatio;
}


double Learn::ArmlearnEvaluationResult::getNbActivatedAction() const{
    return this->nbActivatedAction;
}

double Learn::ArmlearnEvaluationResult::getNbCollision() const{
    return this->nbCollision;
}

std::vector<std::pair<int, double>> Learn::ArmlearnEvaluationResult::getTrajScores(){
    return this->trajScores;
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

        // Weighted addition of propActivatedRoots
        this->nbActivatedTeam = this->nbActivatedTeam * (double)this->nbEvaluation +
                       otherConverted.nbActivatedTeam * (double)otherConverted.nbEvaluation;
        this->nbActivatedTeam /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Weighted addition of propActivatedRoots
        this->nbActivatedTeamRatio = this->nbActivatedTeamRatio * (double)this->nbEvaluation +
                       otherConverted.nbActivatedTeamRatio * (double)otherConverted.nbEvaluation;
        this->nbActivatedTeamRatio /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Weighted addition of propActivatedRoots
        this->nbActivatedAction = this->nbActivatedAction * (double)this->nbEvaluation +
                       otherConverted.nbActivatedAction * (double)otherConverted.nbEvaluation;
        this->nbActivatedAction /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;


        this->nbCollision = this->nbCollision * (double)this->nbEvaluation +
                       otherConverted.nbCollision * (double)otherConverted.nbEvaluation;
        this->nbCollision /= (double)this->nbEvaluation + (double)otherConverted.nbEvaluation;

        // Addition ot nbEvaluation
        this->nbEvaluation += otherConverted.nbEvaluation;
    }

    return *this;
}


