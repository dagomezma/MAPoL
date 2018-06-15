#ifndef LOADMATPOWERFORMAT_H_
#define LOADMATPOWERFORMAT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <thrust/host_vector.h>
#include <cmath>

#include <powersystem/Topology.h>

Topology loadMatpowerFormat(std::string path) {
    std::ifstream dataset;
    dataset.open(path.c_str());

    if (dataset.fail()) {
        std::cout << "Falha ao abrir caso de teste!" << std::endl;
        std::exit(1);
    }

    thrust::host_vector<unsigned int> busesID;
    thrust::host_vector<Bus> buses;
    thrust::host_vector<Branch> branches;
    thrust::host_vector<Generator> gens;
    double baseMVA;
    std::string line;
    getline(dataset, line);

    while (dataset >> line) {
        if (line[0] == '%' || line.empty()) {
            getline(dataset, line);
            continue;
        }
        if (line.compare("mpc.version") == 0) {
            dataset >> line;
            dataset >> line;
            if (line.compare("'2';") != 0) {
                std::cout << "Input MathPower case old version!" << std::endl;
                std::exit(1);
            }
        }
        if (line.compare("mpc.baseMVA") == 0) {
            dataset >> line;
            dataset >> baseMVA;
            getline(dataset, line);
        }
        if (line.compare("mpc.bus") == 0) {
            getline(dataset, line);
            dataset >> line;
            while (line.compare("];") != 0) {
                unsigned int id; // ID da barra
                Bus::Type type;      // tipo de barra
                double V;       // tensão na barra (p.u.)
                double Vmax;    // tensão máxima (p.u.)
                double Vmin;    // tensão mínima (p.u.)
                double baseKV;   // tensão base (kV)
                double O;       // ângulo da tensão (graus)
                double P;       // potência ativa demandada (MW)
                double Q;       // potência reativa demandada (MVAr)
                double Gsh;     // condutância shunt (p.u.)
                double Bsh;     // susceptância shunt (p.u.)

                double tmp_d;
                id = std::atof(line.c_str());
                dataset >> tmp_d;
                type = Bus::Type(tmp_d);
                dataset >> P;
                dataset >> Q;
                dataset >> Gsh;
                dataset >> Bsh;
                dataset >> tmp_d;
                dataset >> V;
                dataset >> O;
                dataset >> baseKV;
                dataset >> tmp_d;
                dataset >> Vmax;
                dataset >> Vmin;
                buses.push_back(
                        Bus(busesID.size(), type, V, O * M_PI / 180.0, 0.0, -P / baseMVA, -Q / baseMVA, Gsh / baseMVA,
                                Bsh / baseMVA, Vmax, Vmin, baseKV, -1, -1));
                busesID.push_back(id);
                getline(dataset, line);
                dataset >> line;
            }
        }
        if (line.compare("mpc.gen") == 0) {
            getline(dataset, line);
            dataset >> line;
            while (line.compare("];") != 0) {
                unsigned int busID; // ID da barra geradora
                double P = 0;           // potência ativa gerada
                double Q = 0;           // potência reativa gerada
                double Pmax;        // máxima potência ativa gerada
                double Pmin;        // mínima potência ativa gerada
                double Qmax;        // máxima potência reativa gerada
                double Qmin;        // mínima potência reativa gerada
                double VG;        // Tensão do Gerador

                double tmp_d;
                busID = std::atof(line.c_str());
                for (int i = 0; i < busesID.size(); i++) {
                    if (busID == busesID[i]) {
                        busID = i;
                        break;
                    }
                }
                dataset >> P;
                buses[busID].P += P / baseMVA;
                dataset >> Q;
                buses[busID].Q += Q / baseMVA;
                dataset >> Qmax;
                dataset >> Qmin;
                dataset >> VG;
                buses[busID].VG = VG;
                dataset >> tmp_d;
                dataset >> tmp_d;
                dataset >> Pmax;
                dataset >> Pmin;
                gens.push_back(Generator(busID, P, Q, Pmax, Pmin, Qmax, Qmin));
                getline(dataset, line);
                dataset >> line;
            }
        }
        if (line.compare("mpc.branch") == 0) {
            getline(dataset, line);
            dataset >> line;
            while (line.compare("];") != 0) {
                unsigned int from;  // barra origem
                unsigned int to;    // barra destino
                bool inservice;     // se o circuito está fechado
                double R;           // resistência (p.u.)
                double X;           // reatância (p.u.)
                double B;           // susceptância da linha (p.u.)
                double tap;        // tap do transformador (tap = from.V / to.V)
                double shift;       // defasagem em transformador defasador
//                double Pfrom;       // potência ativa injetada em from
//                double Qfrom;       // potência reativa injetada em from
//                double Pto;         // potência ativa injetada em to
//                double Qto;         // potência reativa injetada em to

                double tmp_d;
                from = std::atof(line.c_str());
                dataset >> to;
                dataset >> R;
                dataset >> X;
                dataset >> B;
                dataset >> tmp_d;
                dataset >> tmp_d;
                dataset >> tmp_d;
                dataset >> tap;
                dataset >> shift;
                dataset >> inservice;
                bool fromOK = false;
                bool toOK = false;
                for (int i = 0; i < busesID.size(); ++i) {
                    if (!fromOK && busesID[i] == from) {
                        from = i;
                        fromOK = true;
                    }
                    if (!toOK && busesID[i] == to) {
                        to = i;
                        toOK = true;
                    }
                    if (fromOK && toOK) {
                        break;
                    }
                }
                branches.push_back(
                        Branch(from, to, R, X, B, tap, shift, inservice));
                getline(dataset, line);
                dataset >> line;
            }
        }
    }
    Topology topo(baseMVA, busesID, buses, branches, gens);
    return topo;
}

#endif /* LOADMATPOWERFORMAT_H_ */
