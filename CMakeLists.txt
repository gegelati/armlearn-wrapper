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

if(WIN32)
    set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
    # find the gegelatilib-x.y.z folder in the lib directory.
    file(GLOB GEGELATI_ROOT_DIR "${LIBS_DIR}/gegelatilib-[\\.|0-9]*")
    set(ENV{GEGELATI_DIR} ${GEGELATI_ROOT_DIR})
endif()
find_package(GEGELATI)


if (WIN32)
    file(GLOB
            GEGELATI_DLL
            ${GEGELATI_ROOT_DIR}/bin/*.dll
            )

    MESSAGE("Copy GEGELATI DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
    endif()
endif()



# *******************************************
# ************** Executables ****************
# *******************************************

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib/libtorch")
find_package(Torch)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainSoftActorCritic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainMultipleTraining.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/graphPruner.cpp)

# Delete softActorCritic files for know
file(GLOB 
        sacFiles
        ./src/softActorCritic/*.cpp
        ./src/softActorCritic/*.h)

foreach(sacFile ${sacFiles})
    list(REMOVE_ITEM
            armgegelati_files
            ${sacFile})
endforeach()


# includes JsonCpp library to be able to parse parameters JSON files
include_directories (/usr/local/include/eigen3)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/graphPruner.cpp)


# armMultipleTraining: Create Graph Pruner
add_executable(armGraphPruner ${armgegelati_files})
target_link_libraries(armGraphPruner ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_compile_definitions(armGraphPruner PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")


# armGrabGegelati: Control the robotic arm with gegelati.
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/graphPruner.cpp)
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


list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.h)

list(APPEND
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainSoftActorCritic.cpp)

foreach(sacFile ${sacFiles})
    list(APPEND
            armgegelati_files
            ${sacFile})
endforeach()

# armSoftActorCritic: Training SoftActorCritic on the ArmLearnWrapper learning environment
add_executable(armSoftActorCritic ${armgegelati_files})

target_link_libraries(armSoftActorCritic ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so ${TORCH_LIBRARIES})
target_compile_definitions(armSoftActorCritic PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")



# *******************************************
# *********** Directory creation ************
# *******************************************

if(NOT EXISTS ${CMAKE_BINARY_DIR}/params)
        add_custom_command(
                TARGET ${PROJECT_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/params
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/params.json ${CMAKE_BINARY_DIR}/params/params.json
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/trainParams.json ${CMAKE_BINARY_DIR}/params/trainParams.json
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/sacParams.json ${CMAKE_BINARY_DIR}/params/sacParams.json
                COMMENT "Creating params directory and copying parameters' files")
endif()

if(NOT EXISTS ${CMAKE_BINARY_DIR}/outLogs)
        add_custom_command(
                TARGET ${PROJECT_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/outLogs
                COMMENT "Creating outLogs directory")
endif()

if(NOT EXISTS ${CMAKE_BINARY_DIR}/outLogs/models)
        add_custom_command(
                TARGET ${PROJECT_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/outLogs/models
                COMMENT "Creating outLogs/models directory")
endif()
