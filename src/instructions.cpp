#define _USE_MATH_DEFINES // To get M_PI
#include <math.h>

#include "instructions.h"
#include "approximateComputingTools.h"

void fillInstructionSet(Instructions::Set& set, TrainingParameters params) {

    const double scaleFactor = params.scaleFactor;
    
    if (params.instrType == "double") {
        auto add = [scaleFactor](double a, double b) -> double { return (a * scaleFactor) + (b * scaleFactor); };
        auto minus = [scaleFactor](double a, double b) -> double { return (a * scaleFactor) - (b * scaleFactor); };
        auto times = [scaleFactor](double a, double b) -> double { return (a * scaleFactor) * (b * scaleFactor); };
        auto divide = [scaleFactor](double a, double b) -> double { return (a * scaleFactor) / (b * scaleFactor); };
        auto max = [scaleFactor](double a, double b) -> double { return std::max((a * scaleFactor), (b * scaleFactor)); };
        auto cos = [scaleFactor](double a) -> double { return std::cos(a * scaleFactor); };
        auto sin = [scaleFactor](double a) -> double { return std::sin(a * scaleFactor); };
        auto tan = [scaleFactor](double a) -> double { return std::tan(a * scaleFactor); };
        auto exp = [scaleFactor](double a) -> double { return std::exp(a * scaleFactor); };
        auto log = [scaleFactor](double a) -> double { return std::log(a * scaleFactor); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = max($1, $2);")));
        
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
        auto add = [scaleFactor](double a, double b) -> double { return (float)(a * scaleFactor) + (float)(b * scaleFactor); };
        auto minus = [scaleFactor](double a, double b) -> double { return (float)(a * scaleFactor) - (float)(b * scaleFactor); };
        auto times = [scaleFactor](double a, double b) -> double { return (float)(a * scaleFactor) * (float)(b * scaleFactor); };
        auto divide = [scaleFactor](double a, double b) -> double { return (float)(a * scaleFactor) / (float)(b * scaleFactor); };
        auto max = [scaleFactor](double a, double b) -> double { return std::max((float)(a * scaleFactor), (float)(b * scaleFactor)); };
        auto cos = [scaleFactor](double a) -> double { return std::cos((float)(a * scaleFactor)); };
        auto sin = [scaleFactor](double a) -> double { return std::sin((float)(a * scaleFactor)); };
        auto tan = [scaleFactor](double a) -> double { return std::tan((float)(a * scaleFactor)); };
        auto exp = [scaleFactor](double a) -> double { return std::exp((float)(a * scaleFactor)); };
        auto log = [scaleFactor](double a) -> double { return std::log((float)(a * scaleFactor)); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = max($1, $2);")));
        
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
        auto add = [scaleFactor](double a, double b) -> double { return (int)(a * scaleFactor) + (int)(b * scaleFactor); };
        auto minus = [scaleFactor](double a, double b) -> double { return (int)(a * scaleFactor) - (int)(b * scaleFactor); };
        auto times = [scaleFactor](double a, double b) -> double { return (int)(a * scaleFactor) * (int)(b * scaleFactor); };
        auto divide = [scaleFactor](double a, double b) -> double { return f_div((int)(a * scaleFactor), (int)(b * scaleFactor)); };
        auto max = [scaleFactor](double a, double b) -> double { return std::max((int)(a * scaleFactor), (int)(b * scaleFactor)); };
        auto cos = [scaleFactor](double a) -> double { return std::cos((int)(a * scaleFactor)); };
        auto sin = [scaleFactor](double a) -> double { return std::sin((int)(a * scaleFactor)); };
        auto tan = [scaleFactor](double a) -> double { return std::tan((int)(a * scaleFactor)); };
        auto exp = [scaleFactor](double a) -> double { return f_pow2((int)(a * scaleFactor)); };
        auto log = [scaleFactor](double a) -> double { return f_log2((int)(a * scaleFactor)); };

        set.add(*(new Instructions::LambdaInstruction<double, double>(add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus, "$0 = $1 - $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(times, "$0 = $1 * $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(divide, "$0 = $1 / $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(max, "$0 = max($1, $2);")));
        
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
