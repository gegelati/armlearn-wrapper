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
	auto exp = [](double a)->double {return std::exp(a); };


    set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
    set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
    set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
    set.add(*(new Instructions::LambdaInstruction<double, double>(cond, "$0 = (($1) < ($2)) ? -1*($1) : ($1);")));
    
    set.add(*(new Instructions::LambdaInstruction<double>(cos, "$0 = cos($1);")));
    
    if(params.useInstrSinLn){
        auto sin = [](double a) -> double { return std::sin(a); };
        auto ln = [](double a)->double {return std::log(a); };

        set.add(*(new Instructions::LambdaInstruction<double>(sin, "$0 = sin($1);")));
        set.add(*(new Instructions::LambdaInstruction<double>(ln, "$0 = log($1);")));
    }

    set.add(*(new Instructions::LambdaInstruction<double>(exp, "$0 = exp($1);")));
	

    if(params.useInstrConst){
        auto multByConst = [](Data::Constant c) -> double { return (double)c ; };
        set.add(*(new Instructions::LambdaInstruction<Data::Constant>(multByConst, "$0 = $1;")));
    }

    if(params.useInstrDist2d){
        auto dist2d = [](double a, double b, double c, double d)->double { return std::sqrt(std::pow(a - b, 2) + std::pow(c - d, 2)); };
        set.add(*(new Instructions::LambdaInstruction<double, double, double, double>(dist2d,
        "$0 = sqrt(pow($1 - $2,2) + pow($3 - $4,2));")));
    }

    if(params.useInstrDist3d){
        auto dist3d = [](double a, double b, double c, double d, double e, double f)->double {return std::sqrt(
            std::pow(a - b, 2) + std::pow(c - d, 2) + std::pow(e - f, 2)); };
        set.add(*(new Instructions::LambdaInstruction<double, double, double, double, double, double>(dist3d,
        "$0 = sqrt(pow($1 - $2,2) + pow($3 - $4,2)) + pow($5 - $6,2));")));
    }

    if(params.useInstrSphericalCoordRad){
        auto spherical_rad = [](double a, double b)-> double { return a * a + b * b; };
        set.add(*(new Instructions::LambdaInstruction<double, double>(spherical_rad,
        "$0 = pow($1,2) + pow($2,2);")));
    }

    if(params.useInstrSphericalCoordAngle){
        auto spherical_angle = [](double a, double b)-> double { return std::atan(a / b); };
        set.add(*(new Instructions::LambdaInstruction<double, double>(spherical_angle, "$0 = atan($1 / $2);")));
    }

    if(params.useInstrPi){
	    auto pi = [](double a) -> double { return M_PI; };
        set.add(*(new Instructions::LambdaInstruction<double>(pi, "$0 = M_PI;")));
    }

    if(params.useInstrSquare){
	    auto square = [](double a) -> double { return std::pow(a,2); };
        set.add(*(new Instructions::LambdaInstruction<double>(square, "$0 = pow($1,2);")));
    }

    if(params.useInstrSquareRoot){
	    auto square = [](double a) -> double { return std::sqrt(a); };
        set.add(*(new Instructions::LambdaInstruction<double>(square, "$0 = sqrt($1);")));
    }
}