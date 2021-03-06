#include <iostream>
#include <armlearn/serialcontroller.h>
#include <armlearn/widowxbuilder.h>
#include <armlearn/trajectory.h>
#include <armlearn/nowaitarmsimulator.h>
#include <armlearn/optimcartesianconverter.h>

int maine() {
    /******************************************/
    /****     Common base for examples     ****/
    /******************************************/

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


    /******************************************/
    /****       Grab object example        ****/
    /******************************************/

    /*
     * Set arm to backhoe position. Grab and move an invisible object located at its right in front of him and drop it. Go to sleep position.
     *
     */

    armlearn::Trajectory path(&arbotix);
/*
    path.addPoint(BACKHOE_POSITION);
    path.addPoint({1024, 2200, 2200, 1025, 512, 511});
    path.addPoint({1024, 2400, 2200, 1200, 512, 511});
    path.addPoint({1024, 2400, 2200, 1200, 512, 135});
    path.addPoint({1024, 2200, 2200, 1025, 512, 135});
    path.addPoint({2048, 2200, 2200, 1025, 512, 135});
    path.addPoint({2048, 2400, 2200, 1200, 512, 135});
    path.addPoint({2048, 2400, 2200, 1200, 512, 511});
    path.addPoint({2048, 2200, 2200, 1025, 512, 511});*/
    path.addPoint(SLEEP_POSITION);

    path.printTrajectory();


    path.init();
    path.executeTrajectory();
}
