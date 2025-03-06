/**
 * File generated with GEGELATI v1.3.1
 * On the 2025-03-06 13:33:06
 * With the CodeGen::TPGGenerationEngine.
 */

#include "codeGenArmlearn.h"
#include "codeGenArmlearn_program.h"
#include <limits.h>
#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

int bestProgram(double *results, int nb) {
	int bestProgram = 0;
	double bestScore = (isnan(results[0]))? -INFINITY : results[0];
	for (int i = 1; i < nb; i++) {
		double challengerScore = (isnan(results[i]))? -INFINITY : results[i];
		if (challengerScore >= bestScore) {
			bestProgram = i;
			bestScore = challengerScore;
		}
	}
	return bestProgram;
}

enum vertices {T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, A23, A24, A25, A26, A27, A28, A29, A30, };

int inferenceTPG() {
	enum vertices currentVertex = T22;
	while(1) {
		switch (currentVertex) {
		case T0: {
			const enum vertices next[1] = { A24,  };

			double T0Scores[1];

			T0Scores[0] = P0();

			int best = bestProgram(T0Scores, 1);
			currentVertex = next[best];
			break;
		}
		case T1: {
			const enum vertices next[2] = { A25, A23,  };

			double T1Scores[2];

			T1Scores[0] = P1();
			T1Scores[1] = P2();

			int best = bestProgram(T1Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T2: {
			const enum vertices next[2] = { A28, A29,  };

			double T2Scores[2];

			T2Scores[0] = P3();
			T2Scores[1] = P4();

			int best = bestProgram(T2Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T3: {
			const enum vertices next[1] = { A30,  };

			double T3Scores[1];

			T3Scores[0] = P5();

			int best = bestProgram(T3Scores, 1);
			currentVertex = next[best];
			break;
		}
		case T4: {
			const enum vertices next[2] = { A27, A26,  };

			double T4Scores[2];

			T4Scores[0] = P6();
			T4Scores[1] = P7();

			int best = bestProgram(T4Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T5: {
			const enum vertices next[3] = { A27, A26, A25,  };

			double T5Scores[3];

			T5Scores[0] = P8();
			T5Scores[1] = P7();
			T5Scores[2] = P9();

			int best = bestProgram(T5Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T6: {
			const enum vertices next[4] = { T4, T2, A30, A27,  };

			double T6Scores[4];

			T6Scores[0] = P10();
			T6Scores[1] = P11();
			T6Scores[2] = P12();
			T6Scores[3] = P13();

			int best = bestProgram(T6Scores, 4);
			currentVertex = next[best];
			break;
		}
		case T7: {
			const enum vertices next[4] = { T5, A29, A25, A26,  };

			double T7Scores[4];

			T7Scores[0] = P14();
			T7Scores[1] = P15();
			T7Scores[2] = P1();
			T7Scores[3] = P16();

			int best = bestProgram(T7Scores, 4);
			currentVertex = next[best];
			break;
		}
		case T8: {
			const enum vertices next[3] = { T7, T6, A26,  };

			double T8Scores[3];

			T8Scores[0] = P17();
			T8Scores[1] = P18();
			T8Scores[2] = P19();

			int best = bestProgram(T8Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T9: {
			const enum vertices next[4] = { T6, A26, A27, A30,  };

			double T9Scores[4];

			T9Scores[0] = P20();
			T9Scores[1] = P21();
			T9Scores[2] = P22();
			T9Scores[3] = P5();

			int best = bestProgram(T9Scores, 4);
			currentVertex = next[best];
			break;
		}
		case T10: {
			const enum vertices next[5] = { T2, A26, T6, A30, T9,  };

			double T10Scores[5];

			T10Scores[0] = P11();
			T10Scores[1] = P21();
			T10Scores[2] = P18();
			T10Scores[3] = P23();
			T10Scores[4] = P24();

			int best = bestProgram(T10Scores, 5);
			currentVertex = next[best];
			break;
		}
		case T11: {
			const enum vertices next[7] = { T9, T10, T8, T1, T2, T4, T7,  };

			double T11Scores[7];

			T11Scores[0] = P25();
			T11Scores[1] = P26();
			T11Scores[2] = P27();
			T11Scores[3] = P28();
			T11Scores[4] = P29();
			T11Scores[5] = P30();
			T11Scores[6] = P31();

			int best = bestProgram(T11Scores, 7);
			currentVertex = next[best];
			break;
		}
		case T12: {
			const enum vertices next[2] = { T3, T11,  };

			double T12Scores[2];

			T12Scores[0] = P32();
			T12Scores[1] = P33();

			int best = bestProgram(T12Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T13: {
			const enum vertices next[1] = { T12,  };

			double T13Scores[1];

			T13Scores[0] = P34();

			int best = bestProgram(T13Scores, 1);
			currentVertex = next[best];
			break;
		}
		case T14: {
			const enum vertices next[2] = { T13, T4,  };

			double T14Scores[2];

			T14Scores[0] = P35();
			T14Scores[1] = P36();

			int best = bestProgram(T14Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T15: {
			const enum vertices next[2] = { T4, T13,  };

			double T15Scores[2];

			T15Scores[0] = P37();
			T15Scores[1] = P38();

			int best = bestProgram(T15Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T16: {
			const enum vertices next[2] = { T4, T13,  };

			double T16Scores[2];

			T16Scores[0] = P37();
			T16Scores[1] = P39();

			int best = bestProgram(T16Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T17: {
			const enum vertices next[3] = { T15, T0, T4,  };

			double T17Scores[3];

			T17Scores[0] = P40();
			T17Scores[1] = P41();
			T17Scores[2] = P30();

			int best = bestProgram(T17Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T18: {
			const enum vertices next[1] = { T14,  };

			double T18Scores[1];

			T18Scores[0] = P42();

			int best = bestProgram(T18Scores, 1);
			currentVertex = next[best];
			break;
		}
		case T19: {
			const enum vertices next[3] = { T15, T0, T18,  };

			double T19Scores[3];

			T19Scores[0] = P40();
			T19Scores[1] = P41();
			T19Scores[2] = P43();

			int best = bestProgram(T19Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T20: {
			const enum vertices next[2] = { T11, T16,  };

			double T20Scores[2];

			T20Scores[0] = P44();
			T20Scores[1] = P45();

			int best = bestProgram(T20Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T21: {
			const enum vertices next[3] = { T20, T19, T17,  };

			double T21Scores[3];

			T21Scores[0] = P46();
			T21Scores[1] = P47();
			T21Scores[2] = P48();

			int best = bestProgram(T21Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T22: {
			const enum vertices next[1] = { T21,  };

			double T22Scores[1];

			T22Scores[0] = P49();

			int best = bestProgram(T22Scores, 1);
			currentVertex = next[best];
			break;
		}
		case A23: {
			return 1;
			break;
		}
		case A24: {
			return 2;
			break;
		}
		case A25: {
			return 4;
			break;
		}
		case A26: {
			return 5;
			break;
		}
		case A27: {
			return 6;
			break;
		}
		case A28: {
			return 7;
			break;
		}
		case A29: {
			return 8;
			break;
		}
		case A30: {
			return 0;
			break;
		}
		}
	}
}
