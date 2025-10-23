# armlearn-wrapper
The purpose of this repository is to link the TPG [GEGELATI library](https://github.com/gegelati/gegelati) to the [armlearn library](https://github.com/ggendro/armlearn). 

## How to install ?
First of all, clone the repository and cd in it:
```
$ git clone https://github.com/gegelati/armlearn-wrapper.git
$ cd armlearn-wrapper
```

### Classic installation

You now need to get the dependencies of this project.
On linux all you need to do is to go in the main folder and run the following:
```
$ scripts/dependencies_installation.sh
```
It will download dependencies, put them in a "lib" folder and install them.
It could take a while.

Once it is done, you can build and execute the application:
```
$ mkdir build && cd build && cmake .. && cmake --build .
$ Release/armlearn-wrapper
```

### Singularity image

If you want to build the project with singularity, you can directly use the **buildSingularity.def** file in the script folder

```
$ singularity build image.sif buildSingularity.def
```

You may need folders to save the logs, then you can launch the image, with the logs folder binded

```
$ mkdir outLogs && mkdir outLogs/dotfiles
$ singularity run --bind outLogs/:GRETSI2025-Artifacts/armlearn-wrapper/outLogs image.sif
```


## How does this work ?
The armlearn-wrapper is an application using a Gegelati learner on an armlearn task. Gegelati provides a way to generate and train TPG (agents), and armlearn handles the arm simulation during the evaluation.  

## About the learning strategies

A complementary Readme has been written to help the reader understand the learning strategies for training the AI agents. 
Informations include: 
- Objective of the learning process
- Possible initial positions
- Possible target positions 
- Input values fed to the AI agent by the learning environment (states)
- Output values transmitted from the AI agent to the learning environment (action)
- fitness function (reward shapping)
- Methods for evaluation
README-LEARNING-STRATEGY.md

## Learning Environment supports multiple types  

The Learning Environment can be interfaced with the following types when performing inference **{double, float, int, fixedpt}**. 
Its up to you to **provide a TPG that uses those types** to do its internal computation. TPG code should be inserted in src/codeGen. 

To **compile the Learning Environment for a specific type**, a preprocessor directive must be used. CMake can therefore be configured to pass this directive at compile time. 
```
$cp -r my/tpg/codeGen/codeGenArmlearn* src/codeGen/.
$mkdir build & &cd build
$CXXFLAGS="-DUSE_DFIXEDPT" CFLAGS="-DUSE_FIXEDPT" cmake ..
$make armCodeGen
```


## License
This project is distributed under the CeCILL-C license (see LICENSE file).
