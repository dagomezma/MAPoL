/*
 * solucao.h
 *
 *  Created on: 04/03/2016
 *      Author: vincent
 */

#ifndef PSO_SOLUCAO_H_
#define PSO_SOLUCAO_H_

#include <ctime>
#include <vector>
#include <string>
#include <sstream>

namespace pso {

class Solucao {
	std::vector<double> valores;
	double _fitness;
	double _tempo;
	uint _iteracao;

public:
	Solucao() :
			valores(0), _fitness(0), _tempo(0), _iteracao(0) {
	}

	Solucao(std::vector<double> &valores, double fitness, double tempo,
			uint iteracao) :
			valores(valores), _fitness(fitness), _tempo(tempo), _iteracao(
					iteracao) {
	}

	double operator[](uint posicao) {
		return valores[posicao];
	}

	double fitness() {
		return _fitness;
	}

	double tempo() {
		return _tempo;
	}

	uint iteracao() {
		return _iteracao;
	}

	operator std::string() {
		std::string ret;
		std::stringstream ss;
		ss << "Solucao:" <<
				_fitness << ';' <<
				_tempo << ';' <<
				_iteracao << ';';
		for (int i = 0; i < valores.size(); ++i) {
			ss << valores[i] << ';';
		}
		ss >> ret;
		return ret;
	}
};

}  // namespace pso

#endif /* PSO_SOLUCAO_H_ */
