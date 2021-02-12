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

    // if we want to test the best agent
    if (false) {
        agentTest();
        return 0;
    }

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



    // Set the parameters for the learning process.
    // (Controls mutations probability, program lengths, and graph size
    // among other things)
    // Loads them from "params.json" file
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    int i=0;

    // Instantiate the LearningEnvironment
    ArmLearnWrapper le(&i);

    // Instantiate and init the learning agent
    Learn::ParallelLearningAgent la(le, set, params);
    la.init();

    // Adds a logger to the LA (to get statistics on learning) on std::cout
    /*auto logCout = *new Log::LABasicLogger();
    la.addLogger(logCout);*/

    // Adds another logger that will log in a file
    std::ofstream o("log");
    auto logFile = *new Log::LABasicLogger(la,o);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_000.dot", la.getTPGGraph());


    printf("\nGen\tNbVert\tMin\tAvg\tMax\tTvalid\n");

    armlearn::Input<uint16_t> * randomGoal;
    auto validationGoal = armlearn::Input<int16_t>({300, 100, 100});
    auto startEval = std::chrono::high_resolution_clock::now();

    // Train for NB_GENERATIONS generations
    for (i = 0; i < NB_GENERATIONS; i++) {

        le.targets.clear();

        // we generate random targets so that at each generation, 100 different targets are used.
        // as target changes on reset, make sure nbIterationsPerPolicyEvaluation > 100
        for(int j=0; j<100; j++){
            auto target = le.randomGoal();
            le.targets.emplace_back(target);
        }

        char buff[16];
        sprintf(buff, "out_%03d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();


        la.trainOneGeneration(i);


        // loads the validation goal to get learning stats on a given target
        le.targets.clear();
        le.targets.emplace_back(&validationGoal);


        std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> result;

        result = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);

        auto stopEval = std::chrono::high_resolution_clock::now();
        auto iter = result.begin();
        double min = iter->first->getResult();
        std::advance(iter, result.size() - 1);
        double max = iter->first->getResult();
        double avg = std::accumulate(result.begin(), result.end(), 0.0,
                                     [](double acc,
                                        std::pair<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> pair) -> double {
                                         return acc + pair.first->getResult();
                                     });
        avg /= result.size();
        printf("%3d\t%4" PRIu64 "\t%1.2lf\t%1.2lf\t%1.2lf", i, la.getTPGGraph().getNbVertices(), min, avg, max);
        std::cout << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(stopEval - startEval).count()/1000
                  << std::endl;
    }

    // Keep best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();



    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }


    return 0;
}


