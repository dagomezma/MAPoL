/*
 * grid.h
 *
 *  Created on: 06/03/2016
 *      Author: vincent
 */

#ifndef GRID_H_
#define GRID_H_

#include "../run.h"
#include "../Parametros.h"

namespace tuning {

pso::Parametros grid(std::vector<double> w_max, std::vector<double> w_min,
		std::vector<double> c, std::vector<double> s,
		std::vector<uint> nParticulas, std::vector<uint> nIteracoes,
		std::vector<uint> tamParticula,
		std::vector<pso::Particula::Estrutura> &estrutura) {
	bool begin = true;
	pso::Solucao melhor;
	pso::Parametros ret;
	for (int i1 = 0; i1 < w_max.size(); ++i1) {
		for (int i2 = 0; i2 < w_min.size(); ++i2) {
			for (int i3 = 0; i3 < c.size(); ++i3) {
				for (int i4 = 0; i4 < s.size(); ++i4) {
					for (int i5 = 0; i5 < nParticulas.size(); ++i5) {
						for (int i6 = 0; i6 < nIteracoes.size(); ++i6) {
							for (int i7 = 0; i7 < tamParticula.size(); ++i7) {
								/*
								 * TODO
								 * Colocar o algoritmo pra funcionar.
								 * Preciso implementar uma forma de avaliar qual solução é a melhor
								 */
								std::vector<pso::Solucao> melhores;
								pso::run(&melhores, w_max[i1], w_min[i2], c[i3],
										s[i4], nParticulas[i5], nIteracoes[i6],
										tamParticula[i7], &estrutura);
								pso::Solucao melhorI = melhores[melhores.size()];
								if (melhorI.fitness() < melhor.fitness()
										|| begin) {
									melhor = melhorI;
									pso::Parametros tmp(w_max[i1], w_min[i2],
											c[i3], s[i4], nParticulas[i5],
											nIteracoes[i6], tamParticula[i7]);
									ret = tmp;
								} else if (melhorI.fitness() == melhor.fitness()
										&& melhorI.tempo() < melhor.tempo()) {
									melhor = melhorI;
									pso::Parametros tmp(w_max[i1], w_min[i2],
											c[i3], s[i4], nParticulas[i5],
											nIteracoes[i6], tamParticula[i7]);
									ret = tmp;
								} else if (melhorI.fitness() == melhor.fitness()
										&& melhorI.tempo() == melhor.tempo()
										&& melhorI.iteracao()
												< melhor.iteracao()) {
									melhor = melhorI;
									pso::Parametros tmp(w_max[i1], w_min[i2],
											c[i3], s[i4], nParticulas[i5],
											nIteracoes[i6], tamParticula[i7]);
									ret = tmp;
								}
							}
						}
					}
				}
			}
		}
	}
	return ret;
}

}  // namespace tuning

#endif /* GRID_H_ */
