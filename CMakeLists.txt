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

file(GLOB_RECURSE
        armgegelati_files
        ./src/*.cpp
        ./src/*.h
        )
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/resultTester.h
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/mainGrabStandalone.cpp)

include_directories (/usr/local/include/eigen3)

# armGegelati: Training Gegelati on the ArmLearnWrapper learning environment
add_executable(armGegelati ${armgegelati_files})

target_link_libraries(armGegelati ${GEGELATI_LIBRARIES} /usr/local/lib/libarmlearn.so)
target_link_libraries(armGegelati)
target_compile_definitions(armGegelati PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")


# armGrabGegelati: Control the robotic arm with gegelati.
list(REMOVE_ITEM
        armgegelati_files
        ${CMAKE_CURRENT_SOURCE_DIR}/./src/main.cpp)
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