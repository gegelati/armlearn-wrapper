//
// Created by root on 29/06/2020.
//

#ifndef ARMGEGELATI_RESULTTESTER_H
#define ARMGEGELATI_RESULTTESTER_H

#include <armlearn/input.h>
#include "ArmLearnWrapper.h"

int agentTest();

int runByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le, armlearn::Input<uint16_t>& goal);

int runEvals(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le);

#endif //ARMGEGELATI_RESULTTESTER_H
