cmake_minimum_required(VERSION 3.16)

# **********************
# ArmGegelati project.
#
# Control of the WidowX robotic arm  using the Gegelati library.
#
# This CMake project creates 3 executables:
# - armGegelati: Training Gegelati on the ArmLearnWrapper learning environment
# - armGrabGegelati: Control the robotic arm with gegelati.
# - armGrabStandalone: Control the robotic arm without gegelati.
#
# **********************

project(armGegelati)

set(CMAKE_CXX_STANDARD 17)

if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    # Link with pthread
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    #add libmath during non visual studio builds
    set(CMAKE_EXTRA_LIB m)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add definitions for testing purposes
if(${TESTING})
    MESSAGE("Testing mode")
    add_definitions(-DNB_GENERATIONS=10)
endif()

# Include directories for JSONcpp
include_directories("${CMAKE_SOURCE_DIR}/lib/JsonCpp")

# *******************************************
# *********** GEGELATI LIBRARY **************
# *******************************************

find_package(GEGELATI)




# includes eigen
include_directories (/usr/local/include/eigen3)

# *******************************************
# ************** Executables ****************
# *****************************************

file(GLOB_RECURSE
        armgegelati_files
        ./src/*.cpp
        ./lib/JsonCpp/*.cpp
        ./src/*.h
        )
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainGrabStandalone.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainMultipleTraining.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/graphPruner.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenInf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/externHeader.h)



# armGegelati: Training Gegelati on the ArmLearnWrapper learning environment
add_executable(armGegelati ${armgegelati_files})

target_link_libraries(armGegelati ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armGegelati PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/main.cpp)
list(APPEND
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainMultipleTraining.cpp)


# armMultipleTraining: Create multiple trainings
add_executable(armMultipleTraining ${armgegelati_files})
target_link_libraries(armMultipleTraining ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armMultipleTraining PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainMultipleTraining.cpp)
list(APPEND
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenInf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/externHeader.h)


# armMultipleTraining: Create Graph Pruner
add_executable(armCodeGen ${armgegelati_files})
target_link_libraries(armCodeGen ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armCodeGen PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")




# armGrabGegelati: Control the robotic arm with gegelati.
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.c
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenArmlearn_program.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/codeGenInf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/externHeader.h)
list(APPEND
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/graphPruner.cpp)


# armMultipleTraining: Create Graph Pruner
add_executable(armGraphPruner ${armgegelati_files})
target_link_libraries(armGraphPruner ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armGraphPruner PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")


# armGrabGegelati: Control the robotic arm with gegelati.
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/codeGen/graphPruner.cpp)
list(APPEND
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.h)

add_executable(armGrabGegelati ${armgegelati_files})
target_link_libraries(armGrabGegelati ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armGrabGegelati PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

# armGrabStandalone: Control the robotic arm without gegelati
add_executable(armGrabStandalone ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainGrabStandalone.cpp)
target_link_libraries(armGrabStandalone /usr/local/lib/libarmlearn.so)



# *******************************************
# *********** Directory creation ************
# *******************************************



if(NOT EXISTS ${CMAKE_BINARY_DIR}/../outLogs)
        add_custom_command(
                TARGET ${PROJECT_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/../outLogs
                COMMENT "Creating outLogs directory")
endif()


if(NOT EXISTS ${CMAKE_BINARY_DIR}/../outLogs/dotfiles)
        add_custom_command(
                TARGET ${PROJECT_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/../outLogs/dotfiles
                COMMENT "Creating outLogs/dotfiles directory")
endif()
