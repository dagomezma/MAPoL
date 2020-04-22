/*
 * run.h
 *
 *  Created on: 01/03/2016
 *      Author: vincent
 */

#ifndef PSO_RUN_H_
#define PSO_RUN_H_

#ifdef TEST
#include "../../test/fitness.h"
#define DEBUG
#endif

//#define DEBUG
#include "../util/debug.h"

#include <ctime>
#include <cfloat>
#include "Particula.h"
#include "Solucao.h"

namespace pso {

/**
 * Updates the best global and saves it in the "melhores" vector
 */
inline void atualizaMelhor(std::vector<Solucao> *melhores,
		std::vector<double> *bests_global, double *fitMg, std::vector<Particula> &enxame, uint it) {
	bool mudar = false;
	uint id;
	for (uint i = 0; i < enxame.size(); ++i) {
		/*if (enxame[i].fitMelhor() == DBL_MAX)
			continue;
		uint fitAtual = uint(*fitMg * 1e6);
		uint fitNovo = uint(enxame[i].fitMelhor() * 1e6);*/
		if (enxame[i].fitMelhor() < *fitMg) {
		//if (fitNovo < fitAtual) {
			mudar = true;
			id = i;
			*fitMg = enxame[i].fitMelhor();
		}
	}
	if (mudar) {
		*bests_global = enxame[id].bests;
		Solucao sol(*bests_global, *fitMg, GetTimer(), it);
		melhores->push_back(sol);

		printf("Size = %ld\n", melhores->size());
		printf("Melhor %d: fitness = %f, tempo = %lf\n", it, *fitMg, melhores->at(melhores->size()-1).tempo());
	}
}

/**
 * Executao o PSO, retornando as soluções encontradas a cada atualização do melhor global
 */
void run(std::vector<Solucao> *melhores, double w_max, double w_min, double c,
		double s, uint nParticulas, uint nIteracoes, uint tamParticula,
		std::vector<Particula::Estrutura> *estrutura) {

	double start;
	std::vector<double> bests_global;
	double fitMg;

	// Particle Initialization
	debug("pso::run - Initialization\n");
	std::vector<Particula> enxame(nParticulas);
	enxame[0].inicializar(tamParticula, &bests_global, estrutura, true);
	for (int i = 0; i < enxame[0].X.size(); ++i) {
		debug("X[%d] = %lf\n", i, enxame[0].X[i]);
	}
//	debug("Fitness = %lf\n", mkl_runpf(*estrutura, enxame[0]));
	for (uint i = 1; i < nParticulas; ++i) {
		enxame[i].inicializar(tamParticula, &bests_global, estrutura);
	}

	//TODO Evaluation
	debug("pso::run - Evaluation\n");
#ifdef TEST_FITNESS_H_
	for (uint i = 0; i < nParticulas; ++i) {
		enxame[i].mudarFitness(test::fitness(enxame[i]));
	}
#endif
	switch(execucao){
		case 'S':
		case 'O':
		{
			for (uint i = 0; i < nParticulas; ++i) {
				start = GetTimer();
				enxame[i].mudarFitness(mkl_runpf(*estrutura, enxame[i]));
				timeTable[TIME_RUNPF] += GetTimer() - start;
			}
			break;
		}
		case 'P':
		{
			start = GetTimer();
			hybrid_runpf(*estrutura, enxame);
			timeTable[TIME_RUNPF] += GetTimer() - start;
			break;
		}
		default:
		{
			cout << "Unkown EXEC-TYPE. Valid EXEC-TYPEs: S; P; O" << endl;
			exit(1);
		}
	}

	// Update of the global best
	debug("pso::run - Update of best global\n");
	fitMg = UINT_MAX / 1e6;
	bests_global = enxame[0].bests;  // dgm, why did they do this?
	debug("pso::run - fitMg = %f\n", fitMg);
	atualizaMelhor(melhores, &bests_global, &fitMg, enxame, 0);

	// Main loop
	for (uint it = 1; it <= nIteracoes; ++it) {

		// Movement
		debug("pso::run - Atualização inércia %d\n", it);
		double w = w_max - (w_max - w_min) / nIteracoes * it; // cálculo da inércia


		//TODO Evaluation
#ifdef TEST_FITNESS_H_
		debug("pso::run - Avaliação %d\n", it);
		for (uint i = 0; i < nParticulas; ++i) {
			enxame[i].mudarFitness(test::fitness(enxame[i]));
		}
#endif
		switch(execucao){
			case 'S':
			case 'O':
			{
				debug("pso::run - Movimentação e Avaliação %d\n", it);
				for (uint i = 0; i < nParticulas; ++i) {
					enxame[i].mover(w, c, s); // movimentação da partícula
					start = GetTimer();
					enxame[i].mudarFitness(mkl_runpf(*estrutura, enxame[i]));
					timeTable[TIME_RUNPF] += GetTimer() - start;
				}
				break;
			}
			case 'P':
			{
				debug("pso::run - Movimentação %d\n", it);
				for (uint i = 0; i < nParticulas; ++i) {
					enxame[i].mover(w, c, s); // movimentação da partícula
				}
				debug("pso::run - Avaliação %d\n", it);
				start = GetTimer();
				hybrid_runpf(*estrutura, enxame);
				timeTable[TIME_RUNPF] += GetTimer() - start;
				break;
			}
			default:
			{
				cout << "Comando EXECUCAO desconhecido. Comandos Validos: S e P" << endl;
				exit(1);
			}
		}

		// Update of the global best
		debug("pso::run - Update of the global best %d\n", it);
		atualizaMelhor(melhores, &bests_global, &fitMg, enxame, it);
	}
}

}  // namespace pso

#endif /* PSO_RUN_H_ */
