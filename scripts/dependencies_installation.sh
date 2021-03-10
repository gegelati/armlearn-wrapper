# installation instructions : run the script and go take a coffee.
# If after a some time it fails, the log could provide informations about a missing package. Look for the package in the blocks below to understand what was wrong.

# exit when any command fails
set -e

echo "Beginning of dependencies installation..."
# get basic elements
sudo apt install git make cmake g++ python3 python3-pip python3-catkin python3-catkin-pkg python3-empy python3-nose libgtest-dev libboost-all-dev doxygen libsdl2-image-dev libsdl2-ttf-dev

# dependencies will be put in lib and installed
mkdir -p lib && cd lib

# get Eigen3 (v3.3.9)
echo "# Install Eigen3"
git clone --depth 1 --branch 3.3.9 https://gitlab.com/libeigen/eigen.git
sudo cp -R eigen/Eigen /usr/local/include/

# get kdl (commit 0b1b52e)
echo "# Install orocos_kdl"
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics/orocos_kdl/
git checkout 0b1b52e
mkdir build && cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen
sudo cmake --build . --parallel --target install
cd ../../..

# get catkin (v0.8.9)
echo "# Install catkin"
git clone --depth 1 --branch 0.8.9 https://github.com/ros/catkin.git
cd catkin/bin
cmake ..
sudo cmake --build . --parallel --target install
cd ../..


# get boost (v4.4.0)
echo "# Install boost"
git clone --depth 1 --branch 4.4.0 https://github.com/boostorg/build.git
cd build
./bootstrap.sh
cd ..

# get serial (commit cbcca7)
echo "# Install serial"
git clone https://github.com/wjwwood/serial.git
cd serial
git checkout cbcca7c
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --build . --parallel --target install
cd ../..

# get json from nlohann 
echo "# Install json"
git clone --depth 1 --branch v3.9.1 https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake ..
sudo cmake --build . --parallel --target install
cd ../..

# get armlearn (commit 3987527b)
echo "# Install armlearn"
git clone https://github.com/ggendro/armlearn.git
cd armlearn
git checkout 3987527b
rm -R examples/computations # there is a bad include in it and we don't need it
rm -R examples/device_communication # there is a bad include in it and we don't need it
rm -R examples/learning # there is a bad include in it and we don't need it
rm -R examples/trajectories # there is a bad include in it and we don't need it
mkdir preesm
mkdir preesm/org.ietr.preesm.reinforcement_learning
mkdir preesm/org.ietr.preesm.reinforcement_learning/Spider
mkdir build && cd build
echo "" > ../tests/gtests/CMakeLists.txt # just to avoid building tests as they sometimes don't work and are not compulsory
echo "" > ../preesm/org.ietr.preesm.reinforcement_learning/Spider/CMakeLists.txt #same
cmake ..
sudo cmake --build . --parallel --target install
cd ../..


# get Gegelati (latest version installed, tested with 0.5.1)
echo "# Install Gegelati"
git clone https://github.com/gegelati/gegelati.git
cd gegelati/bin
cmake ..
sudo cmake --build . --parallel --target install # On Linux
cd ../..

# update libs
sudo /sbin/ldconfig -v
echo "Installation finished"

