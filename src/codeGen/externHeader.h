#ifndef EXTERNHEADER
#define EXTERNHEADER

#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../fixedptc/fixedptc.h"

#ifdef USE_INT
  #warning "Using int as typeInf"
  typedef int typeInf;  
#elif defined(USE_FLOAT)
  #warning "Using float as typeInf"
  typedef float  typeInf;
#elif defined(USE_FIXEDPT)
  #warning "Using fixedpt as typeInf"
  typedef fixedpt  typeInf;  // using fixedptc library
#else  // default
  #warning "Using double as typeInf"
  typedef double    typeInf;
#endif

    extern typeInf* in1;
    extern typeInf* in2;
    extern typeInf* in3;
    extern typeInf* in4;

    inline typeInf convEnvToInf(double input){
      #ifdef USE_FIXEDPT
        return double_to_fixedpt(input);
      #endif
        return ((typeInf) input);
    }

#ifdef __cplusplus
}
#endif

#endif // EXTERNHEADER