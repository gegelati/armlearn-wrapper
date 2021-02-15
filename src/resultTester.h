//
// Created by root on 29/06/2020.
//

#ifndef ARMGEGELATI_RESULTTESTER_H
#define ARMGEGELATI_RESULTTESTER_H

#include <armlearn/input.h>
#include "ArmLearnWrapper.h"

int agentTest();

void printPolicyStats(const TPG::TPGVertex* root, Environment& env);

// tests arm on a given pos in step by step mode (user has to press enter between each move) in simulation
void runByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le, armlearn::Input<uint16_t>& goal);

// tests arm on several different targets (eg 1000) in simulation
void runEvals(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le);

// makes real arm reach several positions predefined in the code
void runRealArmAuto(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le);

// makes real arm reach several positions prompted by user
void runRealArmByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le);

// makes simulation arm go to a given target and adds its trajectory in path
void goToPos(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le,
            armlearn::Trajectory& path, armlearn::Input<int16_t> *target);

#endif //ARMGEGELATI_RESULTTESTER_H
