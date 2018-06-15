#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <util/defines.h>

/*
 * Classe que define um gerador
 */
struct Generator {
    unsigned int busID; // ID da barra geradora
    double P;           // potência ativa gerada
    double Q;           // potência reativa gerada
    double Pmax;        // máxima potência ativa gerada
    double Pmin;        // mínima potência ativa gerada
    double Qmax;        // máxima potência reativa gerada
    double Qmin;        // mínima potência reativa gerada

    /*
     * Construtor padrão
     */
    def Generator() :
            busID(0), P(0), Q(0), Pmax(0), Pmin(0), Qmax(0), Qmin(0) {
    }

    /*
     * Construtor de cópia
     */
    def Generator(const Generator& other) :
            busID(other.busID), P(other.P), Q(other.Q), Pmax(other.Pmax), Pmin(
                    other.Pmin), Qmax(other.Qmax), Qmin(other.Qmin) {
    }

    def Generator(unsigned int busID, double realPowerOutput,
            double reactivePowerOutput, double Pmax, double Pmin, double Qmax,
            double Qmin) :
            busID(busID), P(realPowerOutput), Q(reactivePowerOutput), Pmax(
                    Pmax), Pmin(Pmin), Qmax(Qmax), Qmin(Qmin) {
    }
};

#endif /* GENERATOR_H_ */
