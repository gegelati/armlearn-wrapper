#define _USE_MATH_DEFINES // To get M_PI
#include <math.h>

#include "instructions.h"

void fillInstructionSet(Instructions::Set& set, TrainingParameters params) {
    auto minus = [](double a, double b) -> double { return a - b; };
    auto add = [](double a, double b) -> double { return a + b; };
    auto times = [](double a, double b) -> double { return a * b; };
    auto divide = [](double a, double b) -> double { return a / b; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };
    auto cos = [](double a) -> double { return std::cos(a); };
    auto sin = [](double a) -> double { return std::sin(a); };
    
    auto ln = [](double a)->double {return std::log(a); };
	auto exp = [](double a)->double {return std::exp(a); };

    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(times)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(divide)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond)));
    set.add(*(new Instructions::LambdaInstruction<double>(cos)));
    set.add(*(new Instructions::LambdaInstruction<double>(sin)));

    set.add(*(new Instructions::LambdaInstruction<double>(ln)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp)));

    if(params.useInstrDist2d){
        auto dist2d = [](double a, double b, double c, double d)->double {return std::sqrt(std::pow(a - b, 2) + std::pow(c - d, 2)); };
        set.add(*(new Instructions::LambdaInstruction<double, double, double, double>(dist2d)));
    }


    if(params.useInstrDist3d){
        auto dist3d = [](double a, double b, double c, double d, double e, double f)->double {return std::sqrt(
            std::pow(a - b, 2) + std::pow(c - d, 2) + std::pow(e - f, 2)); };
        set.add(*(new Instructions::LambdaInstruction<double, double, double, double, double, double>(dist3d)));
    }


    if(params.useInstrSphericalCoord){
        auto spherical_rad = [](double a, double b)-> double { return a * a + b * b; };
        auto spherical_angle = [](double a, double b)-> double { return std::atan(a / b); };
        set.add(*(new Instructions::LambdaInstruction<double, double>(spherical_rad)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(spherical_angle)));
    }

    

}