/*
 * Parametros.h
 *
 *  Created on: 06/03/2016
 *      Author: vincent
 */

#ifndef PARAMETROS_H_
#define PARAMETROS_H_

#include <vector>

namespace pso {

struct Parametros {
	double w_max;
	double w_min;
	double c;
	double s;
	uint nParticulas;
	uint nIteracoes;
	uint tamParticula;

	Parametros() :
			w_max(0), w_min(0), c(0), s(0), nParticulas(0), nIteracoes(0), tamParticula(
					0) {
	}

	Parametros(double w_max, double w_min, double c, double s, uint nParticulas,
			uint nIteracoes, uint tamParticula) :
			w_max(w_max), w_min(w_min), c(c), s(s), nParticulas(nParticulas), nIteracoes(
					nIteracoes), tamParticula(tamParticula) {
	}

	Parametros(const Parametros& p) :
			w_max(p.w_max), w_min(p.w_min), c(p.c), s(p.s), nParticulas(
					p.nParticulas), nIteracoes(p.nIteracoes), tamParticula(
					p.tamParticula) {
	}

	Parametros& operator=(Parametros& p) {
		w_max = p.w_max;
		w_min = p.w_min;
		c = p.c;
		s = p.s;
		nParticulas = p.nParticulas;
		nIteracoes = p.nIteracoes;
		tamParticula = p.tamParticula;
		return *this;
	}
};

}  // namespace pso

#endif /* PARAMETROS_H_ */
