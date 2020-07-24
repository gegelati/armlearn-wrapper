# armlearn-wrapper
The purpose of this repository is to link the TPG [GEGELATI library](https://github.com/gegelati/gegelati) to the [armlearn library](https://github.com/ggendro/armlearn). 

## How to install ?
First of all, clone the repository and cd in it:
```
$ git clone https://github.com/gegelati/armlearn-wrapper.git
$ cd armlearn-wrapper
```

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

## How does this work ?
The armlearn-wrapper is an application using a Gegelati learner on an armlearn task. Gegelati provides a way to generate and train TPG (agents), and armlearn handles the arm simulation during the evaluation.  

## License
This project is distributed under the CeCILL-C license (see LICENSE file).
