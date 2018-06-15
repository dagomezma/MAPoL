#ifndef BRANCH_H
#define	BRANCH_H

#include <util/defines.h>

/*
 * Classe que define uma conexão entre barras.
 * (e.g. linhas de transmissão, transformadores)
 */
struct Branch {

    /*
     * Atributos
     */
    unsigned int from;  // barra origem
    unsigned int to;    // barra destino
    bool inservice;     // se o circuito está fechado
    double R;           // resistência (p.u.)
    double X;           // reatância (p.u.)
    double B;           // susceptância da linha (p.u.)
    double tap;         // tap do transformador (tap = from.V / to.V)
    double shift;       // defasagem em transformador defasador
    double Pfrom;       // potência ativa injetada em from
    double Qfrom;       // potência reativa injetada em from
    double Pto;         // potência ativa injetada em to
    double Qto;         // potência reativa injetada em to
    int indiceEstrutura;

    /*
     * Construtor padrão.
     * Define valores default para os atributos
     */
    def Branch() :
            from(0), to(0), inservice(false), R(0), X(0), B(0), tap(1), shift(
                    0), Pfrom(0), Qfrom(0), Pto(0), Qto(0), indiceEstrutura(-1){
    }

    /*
     * Construtor de cópia.
     * Cria uma instância com base em outro
     */
    def Branch(const Branch& other) :
            from(other.from), to(other.to), inservice(other.inservice), R(
                    other.R), X(other.X), B(other.B), tap(other.tap), shift(
                    other.shift), Pfrom(other.Pfrom), Qfrom(other.Qfrom), Pto(
                    other.Pto), Qto(other.Qto), indiceEstrutura(other.indiceEstrutura) {
    }

    /*
     * Construtor com atributos.
     * Necessário definir alguns atributos
     */
    def Branch(unsigned int from, unsigned int to, double resistance,
            double reactance, double lineChargingB = 0, double tap = 1,
            double phaseShifter = 0, bool inservice = true, double Pfrom = 0,
            double Qfrom = 0, double Pto = 0, double Qto = 0,int indiceEstrutura = -1) :
            from(from), to(to), inservice(inservice), R(resistance), X(
                    reactance), B(lineChargingB), tap(tap), shift(phaseShifter), Pfrom(
                    Pfrom), Qfrom(Qfrom), Pto(Pto), Qto(Qto), indiceEstrutura(indiceEstrutura) {
    }
};

#endif	/* BRANCH_H */

