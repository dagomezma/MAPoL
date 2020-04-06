/*
 * debug.h
 *
 *  Created on: 26/07/2016
 *      Author: vincent
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#ifdef DEBUG
#include <cstdio>
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...)
#endif


#endif /* DEBUG_H_ */
