/**
* \file approximateComputingTools.h
* \author Paul Allaire
* \license CeCILL-C
* 
* The purpose of this file is to provide approximate computing
* functions for speeding up the inference of the TPG when embedded 
* on the board.
*/

#ifndef APPROXIMATE_COMPUTING_TOOLS_H
#define APPROXIMATE_COMPUTING_TOOLS_H

#include <limits.h>
#include <stddef.h>
#include <inttypes.h>
#include "fixedptc/fixedptc.h"

/* The definition of the inline functions in their header file is intentional
 * to avoid “unresolved external” errors from the linker. That error will occur if 
 * you put the inline function's definition in a . cpp file and if that function is 
 * called from some other */

inline int f_pow2(int si_b) {
  int result;
  if ((si_b < 0) ||
      (si_b > 30) || // si si_b > 30 bits décalage 1<<31 KO | 1<<30 OK
      (1 > (INT_MAX >> si_b))) { //si 1 > (INT_MAX >> si_b) alors si_b est trop grand 
    // Handle error 
    result = 0;
  } else {
    result = 1 << si_b;
  }
  return result;
}


inline int f_log2(int si_a)
{    
  int result = 0;    
	
  if(si_a > 0) {
		for (result = 0; si_a > 1; result++, si_a >>= 1);
	}
  
  return result;
}

// forces the compiler to inline the function even if optimizations are disabled. 
inline int __attribute__((always_inline))f_div(int si_a, int si_b) {
    
  int result = 0;

  if ((si_b == 0) || (si_a == INT_MIN && si_b == -1)) {
          result = 0;
  } else {
    result = si_a / si_b;
  }
  
  return result;
}


#endif
