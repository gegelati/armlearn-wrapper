#include <unordered_set>
#include <string>
#include <cfloat>

#include <gegelati.h>

#include "ArmLearnWrapper.h"
#include "resultTester.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 20000
#endif


int main() {
    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](double a, double b) -> double { return a - b; };
    auto add = [](double a, double b) -> double { return a + b; };
    auto times = [](double a, double b) -> double { return a * b; };
    auto divide = [](double a, double b) -> double { return a / b; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };
    auto cos = [](double a) -> double { return std::cos(a); };
    auto ln = [](double a) -> double { return std::log(a); };
    auto exp = [](double a) -> double { return std::exp(a); };

    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(times)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(divide)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond)));
    set.add(*(new Instructions::LambdaInstruction<double>(cos)));
    set.add(*(new Instructions::LambdaInstruction<double>(ln)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp)));



    // Set the parameters for the learning process.
    // (Controls mutations probability, program lengths, and graph size
    // among other things)
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson("../../params.json",params);

    // Instantiate the LearningEnvironment
    ArmLearnWrapper le;

    // Instantiate and init the learning agent
    Learn::ParallelLearningAgent la(le, set, params);
    la.init();

    // Adds a logger to the LA (to get statistics on learning) on std::cout
    auto logCout = *new Log::LABasicLogger();
    la.addLogger(logCout);

    // Adds another logger that will log in a file
    std::ofstream o("log");
    auto logFile = *new Log::LABasicLogger(o);
    la.addLogger(logFile);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_000.dot", la.getTPGGraph());



    // Train for NB_GENERATIONS generations
    for (int i = 0; i < NB_GENERATIONS; i++) {
        char buff[16];
        sprintf(buff, "out_%03d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();
        std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> result;
        result = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);

        la.trainOneGeneration(i);
    }

    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();



    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    // if we want to test the best agent
    if (true) {
        agentTest();
        return 0;
    }

    return 0;
}


