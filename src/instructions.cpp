#define _USE_MATH_DEFINES // To get M_PI
#include <math.h>

#include "instructions.h"
#include "approximateComputingTools.h"

void fillInstructionSet(Instructions::Set& set, TrainingParameters params) {
    
    if (params.instrType == "double") {
        auto add = [](double a, double b) -> double { return a + b; };
        auto minus = [](double a, double b) -> double { return a - b; };
        auto times = [](double a, double b) -> double { return a * b; };
        auto divide = [](double a, double b) -> double { return a / b; };
        auto max = [](double a, double b) -> double { return std::max(a, b); };
        auto cos = [](double a) -> double { return std::cos(a); };
        auto sin = [](double a) -> double { return std::sin(a); };
        auto tan = [](double a) -> double { return std::tan(a); };
        auto exp = [](double a) -> double { return std::exp(a); };
        auto log = [](double a) -> double { return std::log(a); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        if(params.useInstrExpensiveArithmetic) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
            set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        }
        if(params.useInstrComparison) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = fmax($1, $2);")));
        }
        
        if(params.useInstrTrig) {
            set.add(*(new Instructions::LambdaInstruction<double>(cos, "$0 = cos($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(sin, "$0 = sin($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(tan, "$0 = tan($1);")));
        }

        if(params.useInstrLogExp) {
            set.add(*(new Instructions::LambdaInstruction<double>(log, "$0 = log($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(exp, "$0 = exp($1);")));
        }
    } else if (params.instrType == "float") {
        auto add = [](double a, double b) -> double { return (float)a + (float)b; };
        auto minus = [](double a, double b) -> double { return (float)a - (float)b; };
        auto times = [](double a, double b) -> double { return (float)a * (float)b; };
        auto divide = [](double a, double b) -> double { return (float)a / (float)b; };
        auto max = [](double a, double b) -> double { return std::max((float)a, (float)b); };
        auto cos = [](double a) -> double { return std::cos((float)a); };
        auto sin = [](double a) -> double { return std::sin((float)a); };
        auto tan = [](double a) -> double { return std::tan((float)a); };
        auto exp = [](double a) -> double { return std::exp((float)a); };
        auto log = [](double a) -> double { return std::log((float)a); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        if(params.useInstrExpensiveArithmetic) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
            set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        }
        if(params.useInstrComparison) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = fmaxf($1, $2);")));
        }
        
        if(params.useInstrTrig) {
            set.add(*(new Instructions::LambdaInstruction<double>(cos, "$0 = cos($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(sin, "$0 = sin($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(tan, "$0 = tan($1);")));
        }

        if(params.useInstrLogExp) {
            set.add(*(new Instructions::LambdaInstruction<double>(log, "$0 = log($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(exp, "$0 = exp($1);")));
        }
    } else {
        auto add = [](double a, double b) -> double { return (int)a + (int)b; };
        auto minus = [](double a, double b) -> double { return (int)a - (int)b; };
        auto times = [](double a, double b) -> double { return (int)a * (int)b; };
        auto divide = [](double a, double b) -> double { return f_div((int)a, (int)b); };
        auto max = [](double a, double b) -> double { return std::max((int)a, (int)b); };
        auto cos = [](double a) -> double { return std::cos((int)a); };
        auto sin = [](double a) -> double { return std::sin((int)a); };
        auto tan = [](double a) -> double { return std::tan((int)a); };
        auto exp = [](double a) -> double { return f_pow2((int)a); };
        auto log = [](double a) -> double { return f_log2((int)a); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        if(params.useInstrExpensiveArithmetic) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
            set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        }
        if(params.useInstrComparison) {
            set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = max($1, $2);")));
        }
        
        if(params.useInstrTrig) {
            set.add(*(new Instructions::LambdaInstruction<double>(cos, "$0 = cos($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(sin, "$0 = sin($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(tan, "$0 = tan($1);")));
        }

        if(params.useInstrLogExp) {
            set.add(*(new Instructions::LambdaInstruction<double>(log, "$0 = log($1);")));
            set.add(*(new Instructions::LambdaInstruction<double>(exp, "$0 = exp($1);")));
        }
    }
}
