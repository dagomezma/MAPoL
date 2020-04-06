#ifndef BUS_H
#define	BUS_H

#include "util/defines.h"

/*
 * Classe que define uma barra elétrica
 */
struct Bus {

    /*
     * Enumerador para os tipos de barras elétricas
     */
    enum Type {
        NONE, PQ, PV, SLACK, ISOLATED // Tipo 0 - PQ | 1 - PQ | 2 - PV | 3 - Slack | 4 - isolated
    };

    /*
     * Atributos
     */
    unsigned int id; // ID da barra
    Type type;      // tipo de barra
    double V;       // tensão na barra (p.u.)
    double VG;       // tensão do Gerador (p.u.)
    double Vmax;    // tensão máxima (p.u.)
    double Vmin;    // tensão mínima (p.u.)
    double baseKV;   // tensão base (kV)
    double O;       // ângulo da tensão (graus)
    double P;       // potência ativa demandada (MW)
    double Q;       // potência reativa demandada (MVAr)
    double Gsh;     // condutância shunt (p.u.)
    double Bsh;     // susceptância shunt (p.u.)
    int indicePVPQ;
    int indiceEstrutura;

    /*
     * Construtor padrão.
     * Assume valores default para os atributos
     */
    def Bus() :
            id(0), type(NONE), V(1), VG(1), Vmax(1), Vmin(0), baseKV(1), O(0), P(0), Q(
                    0), Gsh(0), Bsh(0) ,indicePVPQ(-1), indiceEstrutura(-1){
    }

    /*
     * Construtor de cópia
     * Copia uma outra instância da classe Bus
     */
    def Bus(const Bus& other) :
            id(other.id), type(other.type), V(other.V), VG(other.VG), Vmax(other.Vmax), Vmin(
                    other.Vmin), baseKV(other.baseKV), O(other.O), P(other.P), Q(
                    other.Q), Gsh(other.Gsh), Bsh(other.Bsh), indicePVPQ(other.indicePVPQ), indiceEstrutura(other.indiceEstrutura) {
    }

    def Bus(unsigned int id, Type busType, double voltageMagnitude,
            double voltageAngle, double voltageGenerator, double realPower, double reactivePower,
            double shuntCondutance, double shuntSusceptance, double Vmax,
            double Vmin, double baseKV, int indecePvPQ, int indiceEstrutura) :
            id(id), type(busType), V(voltageMagnitude), VG(voltageGenerator), Vmax(Vmax), Vmin(Vmin), baseKV(
                    baseKV), O(voltageAngle), P(realPower), Q(reactivePower), Gsh(
                    shuntCondutance), Bsh(shuntSusceptance),  indicePVPQ(indecePvPQ), indiceEstrutura(indiceEstrutura){
    }
};

#endif	/* BUS_H */

