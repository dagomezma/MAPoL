#ifndef BUS_H
#define	BUS_H

#include <util/defines.h>

struct Bus {
    /*
     * Enum for bus types
     */
    enum Type {
        NONE, PQ, PV, SLACK, ISOLATED // Tipo 0 - PQ | 1 - PQ | 2 - PV | 3 - Slack | 4 - isolated
    };

    /*
     * Attributes
     */
    unsigned int id;  // Bus ID
    Type type;        // Bus type
    double V;         // Bus voltage (p.u.)
    double VG;        // Generator voltage (p.u.)
    double Vmax;      // Max voltage (p.u.)
    double Vmin;      // Min voltage (p.u.)
    double baseKV;    // Base voltage (kV)
    double O;         // Voltage angle (degrees)
    double P;         // Active power demand (MW)
    double Q;         // Reactive power demand (MVAr)
    double Gsh;       // Shunt conductance (p.u.)
    double Bsh;       // Shunt susceptance (p.u.)
    int indicePVPQ;
    int indiceEstrutura;

    /*
     * Default constructor
     * Assumes default values for its attributes
     */
    def Bus() :
            id(0), type(NONE), V(1),
            VG(1), Vmax(1), Vmin(0),
            baseKV(1), O(0), P(0), Q(0),
            Gsh(0), Bsh(0),
            indicePVPQ(-1), indiceEstrutura(-1) {}

    /*
     * Copy operation constructor
     */
    def Bus(const Bus& other) :
            id(other.id), type(other.type), V(other.V),
            VG(other.VG), Vmax(other.Vmax), Vmin(other.Vmin),
            baseKV(other.baseKV), O(other.O), P(other.P), Q(other.Q),
            Gsh(other.Gsh), Bsh(other.Bsh),
            indicePVPQ(other.indicePVPQ), indiceEstrutura(other.indiceEstrutura) {}

    def Bus(unsigned int id, Type busType, double voltageMagnitude,
            double voltageAngle, double voltageGenerator, double realPower,
            double reactivePower, double shuntCondutance, double shuntSusceptance,
            double Vmax, double Vmin, double baseKV,
            int indecePvPQ, int indiceEstrutura) :
            id(id), type(busType), V(voltageMagnitude),
            VG(voltageGenerator), Vmax(Vmax), Vmin(Vmin),
            baseKV(baseKV), O(voltageAngle), P(realPower), Q(reactivePower),
            Gsh(shuntCondutance), Bsh(shuntSusceptance),
            indicePVPQ(indecePvPQ), indiceEstrutura(indiceEstrutura){}
};

#endif	/* BUS_H */

