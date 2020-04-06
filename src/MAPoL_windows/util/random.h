/*
 * random.h
 *
 *  Created on: 09/05/2016
 *      Author: vincent
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <cstdlib>

inline double real_rand(double min = 0, double max = 1) {
	double rd = (double) rand() / RAND_MAX;
	return rd * (max - min) + min;
}

inline int rand(int min, int max) {
	return rand() % (max - min) + min;
}

#endif /* RANDOM_H_ */
