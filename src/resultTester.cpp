#include <iostream>
#include <unordered_set>
#include <numeric>
#include <string>
#include <cfloat>
#include <inttypes.h>
#include <unistd.h>

#include <gegelati.h>

#include "resultTester.h"

#include "ArmLearnWrapper.h"

int main() {
    /*
    // Check sudo rights to connect to the arm
    if (getuid() != 0) {
        std::cerr << "Error: You need to be root to connect to the arm." << std::endl;
        exit(1);
    }
    */

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

    TrainingParameters params;
    ArmLearnWrapper le(1000, params);


    auto * validationGoal = new armlearn::Input<double>({300, 50, 50});
    auto validationStartingPos = le.getInitStartingPos();
    le.customTrajectory(validationGoal, validationStartingPos);
    le.reset();

/*
    // Instantiate the environment that will embed the LearningEnvironment
    Environment env(set, le.getDataSources(), 8);

    // Instantiate the TPGGraph that we will load
    auto tpg = TPG::TPGGraph(env);

    // Instantiate the tee that will handle the decisions taken by the TPG
    TPG::TPGExecutionEngine tee(env);



    // Create an importer for the best graph and imports it
    std::cout << "Import graph"<< std::endl;
    File::TPGGraphDotImporter dotImporter(ROOT_DIR "/dat/best.dot", env, tpg);
    dotImporter.importGraph();

    // takes the first root of the graph, anyway out_best has only 1 root (the best)
    auto root = tpg.getRootVertices().front();

    // make a try on a random position

    std::cout << "Start arm"<< std::endl;
    //runByHand(root, tee, le, validationGoal);

    //runEvals(root,tee,le);

    runRealArmByHand(root,tee,le);

    // runRealArmAuto(root, tee, le);

    //printPolicyStats(root,env);
    */
    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    return 0;
}

void printPolicyStats(const TPG::TPGVertex* root, Environment& env){
    TPG::PolicyStats ps;
    ps.setEnvironment(env);
    ps.analyzePolicy(root);
    std::ofstream bestStats;
    bestStats.open("out_best_stats.md");
    bestStats << ps;
    bestStats.close();
}


void runByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le, armlearn::Input<int16_t>& goal){
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

void runEvals(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le){
    std::cout<<"begining of runEvals"<<std::endl;
    std::ofstream o("log");
    double x=1;
    auto validationStartingPos = le.getInitStartingPos();
    while(x!=100000){
        auto rnd = le.randomGoal(validationStartingPos, true);
        le.customTrajectory(rnd, validationStartingPos);
        le.reset();
        for(int i=0; i<2000; i++) {
            // gets the action the TPG would decide in this situation (the result can only be between 0 and 8 included)
            uint64_t action = ((const TPG::TPGAction *) tee.executeFromRoot(*root).back())->getActionID();
            le.doAction(action);

            // prints the game board
        }
        o << x << " " << le.toString() << std::endl;
        o.flush();
        x++;
    }
    o.close();
}

void runRealArmAuto(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le){

    armlearn::communication::SerialController arbotix("/dev/ttyUSB0");

    armlearn::WidowXBuilder builder;
    builder.buildController(arbotix);

    arbotix.connect();
    std::cout << arbotix.servosToString();


    std::this_thread::sleep_for(
            (std::chrono::milliseconds) 1000); // Usually, a waiting period and a second connect attempt is necessary to reach all devices
    arbotix.connect();
    std::cout << arbotix.servosToString();

    arbotix.changeSpeed(50); // Servomotor speed is reduced for safety

    std::cout << "Update servomotors information:" << std::endl;
    arbotix.updateInfos();

    armlearn::Trajectory path(&arbotix);

    // open pliers here
    auto* goal1 = new armlearn::Input<double>({220, 25, 200});
    auto* goal2 = new armlearn::Input<double>({220, 25, 15});
    // grab here
    auto* goal3 = new armlearn::Input<double>({220, 25, 200});
    auto* goal4 = new armlearn::Input<double>({150, 150, 200});
    auto* goal5 = new armlearn::Input<double>({25, 250, 200});
    auto* goal6 = new armlearn::Input<double>({25, 250, 20});
    // release here
    auto* goal7 = new armlearn::Input<double>({25, 200, 200});


    // open pliers
    auto motorPosOpen = new std::vector<uint16_t>(le.getMotorsPos());
    (*motorPosOpen)[5]=511;
    le.setInitStartingPos(*motorPosOpen);
    path.addPoint(*motorPosOpen);

    goToPos(root, tee, le, path, goal1);
    goToPos(root, tee, le, path, goal2);

    // grab
    auto motorPosGrab = new std::vector<uint16_t>(le.getMotorsPos());
    (*motorPosGrab)[5]=135;
    le.setInitStartingPos(*motorPosGrab);
    path.addPoint(*motorPosGrab);

    goToPos(root, tee, le, path, goal3);
    goToPos(root, tee, le, path, goal4);
    goToPos(root, tee, le, path, goal5);
    goToPos(root, tee, le, path, goal6);

    // release
    auto motorPosRelease = new std::vector<uint16_t>(le.getMotorsPos());
    (*motorPosRelease)[5]=511;
    le.setInitStartingPos(*motorPosRelease);
    path.addPoint(*motorPosRelease);

    goToPos(root, tee, le, path, goal7);

    path.addPoint(SLEEP_POSITION);

    path.printTrajectory();


    path.init();
    path.executeTrajectory();
}

void runRealArmByHand(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le){

    armlearn::communication::SerialController arbotix("/dev/ttyUSB0");

    armlearn::WidowXBuilder builder;
    builder.buildController(arbotix);

    arbotix.connect();
    std::cout << arbotix.servosToString();


    std::this_thread::sleep_for(
            (std::chrono::milliseconds) 1000); // Usually, a waiting period and a second connect attempt is necessary to reach all devices
    arbotix.connect();
    std::cout << arbotix.servosToString();

    arbotix.changeSpeed(50); // Servomotor speed is reduced for safety

    std::cout << "Update servomotors information:" << std::endl;
    arbotix.updateInfos();

    armlearn::Trajectory path(&arbotix);

    // Put in backhoe
    path.addPoint(BACKHOE_POSITION);
    path.printTrajectory();
    path.init();
    path.executeTrajectory();
    path.removePoint();

    double x=0;
    double y=0;
    double z=0;
    while (x!=-1){
        // Get current servo position
        arbotix.updateInfos();
        auto reachedPosition = arbotix.getPosition();
        std::cout << "Servo positions: ";
        for(auto& pos : reachedPosition){
            std::cout << pos << " ";
        }
        std::cout << std::endl;

        // Get current hand position
        armlearn::kinematics::BasicCartesianConverter converter;
        builder.buildConverter(converter);
        auto cartesiandCoords = converter.computeServoToCoord(reachedPosition)->getCoord();
        std::cout << "Hand positions: ";
        for(auto& pos : cartesiandCoords){
            std::cout << pos << " ";
        }
        std::cout << std::endl;

        std::cout<<"x"<<std::endl;
        std::cin>>x;
        if(x==-1){
            break;
        }
        if(x==-2){
            // Put in backhoe
            path.addPoint(BACKHOE_POSITION);
            path.printTrajectory();
            path.init();
            path.executeTrajectory();
            path.removePoint();
            continue;
        }
        std::cout<<"y"<<std::endl;
        std::cin>>y;
        std::cout<<"z"<<std::endl;
        std::cin>>z;

        auto * goal = new armlearn::Input<double>({x, y, z});
        goToPos(root, tee, le, path, goal);

        path.init();
        path.printTrajectory();
        path.executeTrajectory();

        for(int i=0; i<11;i++) {
            path.removePoint();
        }
    }

    path.addPoint(SLEEP_POSITION);

    path.printTrajectory();

    path.init();
    path.executeTrajectory();


}

void goToPos(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, ArmLearnWrapper& le,
            armlearn::Trajectory& path, armlearn::Input<double> *target){
    
    auto validationStartingPos = le.getInitStartingPos();
    le.customTrajectory(target, validationStartingPos);
    le.reset();
    for(int i=0; i<=500; i++) {
        uint64_t action = ((const TPG::TPGAction *) tee.executeFromRoot(*root).back())->getActionID();
        le.doAction(action);
        if (i % 50 == 0) {
            auto motorPos = new const std::vector<uint16_t>(le.getMotorsPos());
            path.addPoint(*motorPos);
        }
    }
    // to avoid going through backhoe
    le.setInitStartingPos(le.getMotorsPos());
}
