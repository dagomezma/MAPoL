/*
 * particula.h
 *
 *  Created on: 01/03/2016
 *      Author: vincent
 */

#ifndef PSO_PARTICULA_H_
#define PSO_PARTICULA_H_

#include <climits>
#include <cfloat>
#include <ctime>
#include <sstream>
//#include <random>
#include <vector>
#include <util/random.h>

namespace pso {

class Particula {
	double _fitness;
	double _fitMelhor;

public:
	struct Estrutura {
		enum Tipo {
			NULO, AVR, OLTC, SHC
		};
		Tipo tipo;
		double min;
		double max;
		double defaultValue;
		uint qtd;
		uint id;

		Estrutura() :
				tipo(NULO), min(-DBL_MAX), max(DBL_MAX), defaultValue(0), qtd(
						0), id(0) {
		}

		Estrutura(Tipo tipo, double min, double max, double def, uint qtd,
				uint id) :
				tipo(tipo), min(min), max(max), defaultValue(def), qtd(qtd), id(
						id) {
		}

		static Estrutura newAVR(double min, double max, double def, uint id) {
			return Estrutura(AVR, min, max, def, 0, id);
		}

		static Estrutura newOLTC(double min, double max, double def, uint qtd,
				uint id) {
			return Estrutura(OLTC, min, max, def, qtd, id);
		}

		static Estrutura newShC(double max, double def, uint qtd, uint id) {
			return Estrutura(SHC, 0, max, def, qtd, id);
		}
	};

	std::vector<double> X;
	std::vector<double> V;
	std::vector<double> M;
	std::vector<double> *Mg;
	std::vector<Estrutura> *estrutura;

	/* Getters */

	size_t size() const {
		return X.size();
	}

	double fitness() const {
		return _fitness;
	}

	double fitMelhor() const {
		return _fitMelhor;
	}

	double operator[](uint posicao) {
		return X[posicao];
	}

	/* Setters */

	void inicializar(uint tamanho, std::vector<double> *Mg,
			std::vector<Estrutura> *estrutura, bool defaultValues = false) {
//		std::default_random_engine gerador(time(0));
//		std::uniform_real_distribution<double> rand(0, 1);

		X.resize(tamanho);
		V.resize(tamanho);
		M.resize(tamanho);
		this->Mg = Mg;
		this->estrutura = estrutura;

		for (uint i = 0; i < tamanho; ++i) {
			V[i] = real_rand(); //rand(gerador);

			if (defaultValues) {
				M[i] = X[i] = estrutura->at(i).defaultValue;
			} else {
				switch (estrutura->at(i).tipo) {
				case Estrutura::AVR: {
//				std::uniform_real_distribution<double> rd(estrutura->at(i).min,
//						estrutura->at(i).max);
					M[i] = X[i] = real_rand(estrutura->at(i).min,
							estrutura->at(i).max); //rd(gerador);
					break;
				}
				case Estrutura::OLTC: {
//				std::uniform_int_distribution<int> rd(0, estrutura->at(i).qtd);
					M[i] = X[i] = rand(0, estrutura->at(i).qtd) //rd(gerador)
					* (estrutura->at(i).max - estrutura->at(i).min)
							/ estrutura->at(i).qtd + estrutura->at(i).min;
					break;
				}
				case Estrutura::SHC: {
//				std::uniform_int_distribution<int> rd(0, estrutura->at(i).qtd);
					M[i] = X[i] = rand(0, estrutura->at(i).qtd) //rd(gerador)
					* estrutura->at(i).max / estrutura->at(i).qtd;
					break;
				}
				default:
					M[i] = X[i] = real_rand(); //rand(gerador);
				}
			}
		}
		_fitMelhor = DBL_MAX;
	}

	void mudarFitness(double valor) {
		_fitness = valor;

		// Atualização do melhor local
		if (_fitness < _fitMelhor) {
			_fitMelhor = _fitness;
			for (uint i = 0; i < X.size(); ++i) {
				M[i] = X[i];
			}
		}
	}

	void mover(double w, double c, double s) {
//		std::default_random_engine gerador(time(0));
//		std::uniform_real_distribution<double> rand(0, 1);

		for (uint i = 0; i < X.size(); ++i) {
			double r1 = real_rand(); //rand(gerador);
			double r2 = real_rand(); //rand(gerador);

			// cálculo da nova velocidade
			V[i] = w * V[i] + r1 * c * (M[i] - X[i])
					+ r2 * s * (Mg->at(i) - X[i]);

			// atualização da posição
			X[i] = X[i] + V[i];
			while (X[i] > estrutura->at(i).max || X[i] < estrutura->at(i).min) {
				if (X[i] > estrutura->at(i).max) {
					X[i] = estrutura->at(i).max - real_rand()/*rand(gerador)*/
					* (X[i] - estrutura->at(i).max);
				} else {
					X[i] = estrutura->at(i).min + real_rand()/*rand(gerador)*/
					* (estrutura->at(i).min - X[i]);
				}
			}
			switch (estrutura->at(i).tipo) {
			case Estrutura::OLTC:
				X[i] = int(
						(X[i] - estrutura->at(i).min) * estrutura->at(i).qtd
								/ (estrutura->at(i).max - estrutura->at(i).min)
								+ 0.5)
						* (estrutura->at(i).max - estrutura->at(i).min)
						/ estrutura->at(i).qtd + estrutura->at(i).min;
				break;
			case Estrutura::SHC:
				X[i] = int(
						X[i] * estrutura->at(i).qtd / estrutura->at(i).max
								+ 0.5) * estrutura->at(i).max
						/ estrutura->at(i).qtd;
				break;
			default:
				break;
			}
		}
	}
};

}  // namespace pso

#endif /* PSO_PARTICULA_H_ */
