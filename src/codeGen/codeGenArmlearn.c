/**
 * File generated with GEGELATI v1.3.1
 * On the 2024-04-17 17:25:05
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

enum vertices {T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, A16, A17, A18, A19, A20, A21, A22, };

int inferenceTPG() {
	enum vertices currentVertex = T15;
	while(1) {
		switch (currentVertex) {
		case T0: {
			const enum vertices next[2] = { A17, A22,  };

			double T0Scores[2];

			T0Scores[0] = P0();
			T0Scores[1] = P1();

			int best = bestProgram(T0Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T1: {
			const enum vertices next[2] = { A19, A16,  };

			double T1Scores[2];

			T1Scores[0] = P2();
			T1Scores[1] = P3();

			int best = bestProgram(T1Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T2: {
			const enum vertices next[6] = { A19, A16, A20, A19, A16, A21,  };

			double T2Scores[6];

			T2Scores[0] = P4();
			T2Scores[1] = P5();
			T2Scores[2] = P6();
			T2Scores[3] = P7();
			T2Scores[4] = P8();
			T2Scores[5] = P9();

			int best = bestProgram(T2Scores, 6);
			currentVertex = next[best];
			break;
		}
		case T3: {
			const enum vertices next[6] = { A19, A16, A19, A16, A21, A20,  };

			double T3Scores[6];

			T3Scores[0] = P4();
			T3Scores[1] = P5();
			T3Scores[2] = P10();
			T3Scores[3] = P8();
			T3Scores[4] = P9();
			T3Scores[5] = P6();

			int best = bestProgram(T3Scores, 6);
			currentVertex = next[best];
			break;
		}
		case T4: {
			const enum vertices next[5] = { A19, A16, A16, A20, A19,  };

			double T4Scores[5];

			T4Scores[0] = P4();
			T4Scores[1] = P5();
			T4Scores[2] = P8();
			T4Scores[3] = P11();
			T4Scores[4] = P12();

			int best = bestProgram(T4Scores, 5);
			currentVertex = next[best];
			break;
		}
		case T5: {
			const enum vertices next[6] = { A20, T2, A16, A19, T1, A16,  };

			double T5Scores[6];

			T5Scores[0] = P6();
			T5Scores[1] = P13();
			T5Scores[2] = P14();
			T5Scores[3] = P15();
			T5Scores[4] = P16();
			T5Scores[5] = P17();

			int best = bestProgram(T5Scores, 6);
			currentVertex = next[best];
			break;
		}
		case T6: {
			const enum vertices next[6] = { A16, A16, A16, T2, A20, A16,  };

			double T6Scores[6];

			T6Scores[0] = P18();
			T6Scores[1] = P19();
			T6Scores[2] = P20();
			T6Scores[3] = P21();
			T6Scores[4] = P22();
			T6Scores[5] = P23();

			int best = bestProgram(T6Scores, 6);
			currentVertex = next[best];
			break;
		}
		case T7: {
			const enum vertices next[2] = { T6, T2,  };

			double T7Scores[2];

			T7Scores[0] = P24();
			T7Scores[1] = P25();

			int best = bestProgram(T7Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T8: {
			const enum vertices next[6] = { A16, T5, A16, A16, T4, A19,  };

			double T8Scores[6];

			T8Scores[0] = P26();
			T8Scores[1] = P27();
			T8Scores[2] = P8();
			T8Scores[3] = P23();
			T8Scores[4] = P28();
			T8Scores[5] = P12();

			int best = bestProgram(T8Scores, 6);
			currentVertex = next[best];
			break;
		}
		case T9: {
			const enum vertices next[2] = { T8, T0,  };

			double T9Scores[2];

			T9Scores[0] = P29();
			T9Scores[1] = P30();

			int best = bestProgram(T9Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T10: {
			const enum vertices next[2] = { T7, T3,  };

			double T10Scores[2];

			T10Scores[0] = P31();
			T10Scores[1] = P32();

			int best = bestProgram(T10Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T11: {
			const enum vertices next[3] = { T3, A16, T0,  };

			double T11Scores[3];

			T11Scores[0] = P32();
			T11Scores[1] = P33();
			T11Scores[2] = P34();

			int best = bestProgram(T11Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T12: {
			const enum vertices next[2] = { A21, T0,  };

			double T12Scores[2];

			T12Scores[0] = P35();
			T12Scores[1] = P36();

			int best = bestProgram(T12Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T13: {
			const enum vertices next[3] = { A18, T10, T9,  };

			double T13Scores[3];

			T13Scores[0] = P37();
			T13Scores[1] = P38();
			T13Scores[2] = P39();

			int best = bestProgram(T13Scores, 3);
			currentVertex = next[best];
			break;
		}
		case T14: {
			const enum vertices next[2] = { T9, T12,  };

			double T14Scores[2];

			T14Scores[0] = P40();
			T14Scores[1] = P41();

			int best = bestProgram(T14Scores, 2);
			currentVertex = next[best];
			break;
		}
		case T15: {
			const enum vertices next[7] = { T0, T14, A16, T9, T13, T5, T11,  };

			double T15Scores[7];

			T15Scores[0] = P42();
			T15Scores[1] = P43();
			T15Scores[2] = P44();
			T15Scores[3] = P45();
			T15Scores[4] = P46();
			T15Scores[5] = P47();
			T15Scores[6] = P48();

			int best = bestProgram(T15Scores, 7);
			currentVertex = next[best];
			break;
		}
		case A16: {
			return 0;
			break;
		}
		case A17: {
			return 1;
			break;
		}
		case A18: {
			return 2;
			break;
		}
		case A19: {
			return 4;
			break;
		}
		case A20: {
			return 5;
			break;
		}
		case A21: {
			return 6;
			break;
		}
		case A22: {
			return 7;
			break;
		}
		}
	}
}
