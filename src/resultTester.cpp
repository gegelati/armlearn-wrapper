#include <iostream>
#include <unordered_set>
#include <numeric>
#include <string>
#include <cfloat>
#include <inttypes.h>

#include <gegelati.h>
#include "resultTester.h"

#include "ArmLearnWrapper.h"

int agentTest() {
    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](double a, double b) -> double { return a - b; };
    auto add = [](double a, double b) -> double { return a + b; };
    auto times = [](double a, double b) -> double { return a * b; };
    auto divide = [](double a, double b) -> double { return a / b; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };
    auto cos = [](double a) -> double { return std::cos(a); };
    auto sin = [](double a) -> double { return std::sin(a); };

    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(times)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(divide)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond)));
    set.add(*(new Instructions::LambdaInstruction<double>(cos)));
    set.add(*(new Instructions::LambdaInstruction<double>(sin)));


    int i=-1;
    ArmLearnWrapper le(&i);
    auto validationGoal = armlearn::Input<uint16_t>({300, 50, 50});
    le.customGoal(&validationGoal);
    le.reset();

    // Instantiate the environment that will embed the LearningEnvironment
    Environment env(set, le.getDataSources(), 8);

    // Instantiate the TPGGraph that we will loead
    auto tpg = TPG::TPGGraph(env);

    // Instantiate the tee that will handle the decisions taken by the TPG
    TPG::TPGExecutionEngine tee(env);



    // Create an importer for the best graph and imports it
    File::TPGGraphDotImporter dotImporter("../Debug/out_355.dot", env, tpg);
    dotImporter.importGraph();

    // takes the first root of the graph, anyway out_best has only 1 root (the best)
    auto root = tpg.getRootVertices().front();

    // make a try on a random position


    //runByHand(root, tee, le, validationGoal);

    runEvals(root,tee,le);

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    return 0;
}


int runByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le, armlearn::Input<uint16_t>& goal){
    double x=1;
    std::cout<<x<<"-Arm :\n"<<le.toString()<<std::endl;
    // let's play, the only way to leave this loop is to enter -1
    while(x!=-1){
        // gets the action the TPG would decide in this situation (the result can only be between 0 and 8 included)
        uint64_t action=((const TPG::TPGAction *) tee.executeFromRoot(* root).back())->getActionID();
        std::cout<<"TPG : "<<action<<std::endl;
        le.doAction(action);

        // prints the game board
        std::cout<<x<<"-Arm :\n"<<le.toString()<<std::endl;
        x++;
        std::cin.ignore();
    }
}

int runEvals(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le){
    std::cout<<"begining of runEvals"<<std::endl;
    double x=1;
    while(x!=1000){
        auto rnd = le.randomGoal();
        le.customGoal(rnd);
        le.reset();
        for(int i=0; i<1000; i++) {
            // gets the action the TPG would decide in this situation (the result can only be between 0 and 8 included)
            uint64_t action = ((const TPG::TPGAction *) tee.executeFromRoot(*root).back())->getActionID();
            le.doAction(action);

            // prints the game board
        }
        std::cout << x << " " << le.toString() << std::endl;
        x++;
    }
}