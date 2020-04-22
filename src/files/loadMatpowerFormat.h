#ifndef LOADMATPOWERFORMAT_H_
#define LOADMATPOWERFORMAT_H_

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <thrust/host_vector.h>
#include <powersystem/Topology.h>

Topology loadMatpowerFormat(std::string path) {
    std::ifstream dataset;
    dataset.open(path.c_str());

    if (dataset.fail()) {
        std::cout << "Failed to open test case!" << std::endl;
        std::exit(1);
    }

    thrust::host_vector<unsigned int> busesID;
    thrust::host_vector<Bus> buses;
    thrust::host_vector<Branch> branches;
    thrust::host_vector<Generator> gens;
    double baseMVA;
    std::string line;
    getline(dataset, line);
    std::cout << "First line: " << line << std::endl;

    int dgm_i = 1;
    while (dataset >> line) {
    	printf("DGM:: inside loadcase function while, i = %d\n", dgm_i);
    	if (line[0] == '%' || line.empty()) {
    		getline(dataset, line);
            continue;
        }
        if (line.compare("mpc.version") == 0) {
        	dataset >> line;
            dataset >> line;
            if (line.compare("'2';") != 0) {
                std::cout << "Matpower case input: old version!" << std::endl;
                std::exit(1);
            }
        }
        if (line.compare("mpc.baseMVA") == 0) {
        	printf("DGM:: inside loadcase, i = %d, now checking baseMVA\n", dgm_i);
        	dataset >> line;
            dataset >> baseMVA;
            getline(dataset, line);
        }
        if (line.compare("mpc.bus") == 0) {
        	printf("DGM:: inside loadcase, i = %d, now checking bus data\n", dgm_i);
        	unsigned int id;  // Bus ID
			Bus::Type type;   // Bus type
			double V;         // Bus voltage (p.u.)
			double Vmax;      // Max voltage (p.u.)
			double Vmin;      // Min voltage (p.u.)
			double baseKV;    // Base voltage (kV)
			double O;         // Voltage angle (degrees)
			double P;         // Active power demand (MW)
			double Q;         // Reactive power demand (MVAr)
			double Gsh;       // Shunt conductance (p.u.)
			double Bsh;       // Shunt susceptance (p.u.)
			double tmp_d;

			getline(dataset, line);
			dataset >> line;

            while (line.compare("];") != 0) {
            	id = std::atof(line.c_str());
                dataset >> tmp_d;
                type = Bus::Type(tmp_d);
                dataset >> P;
                dataset >> Q;
                dataset >> Gsh;
                dataset >> Bsh;
                dataset >> tmp_d;  // area, ignored
                dataset >> V;
                dataset >> O;
                dataset >> baseKV;
                dataset >> tmp_d;  // zone, ignored
                dataset >> Vmax;
                dataset >> Vmin;
                buses.push_back(
                		Bus(busesID.size(), type, V,
                		O * M_PI / 180.0, 0.0, -P / baseMVA, -Q / baseMVA,
                		Gsh / baseMVA, Bsh / baseMVA, Vmax,
                		Vmin, baseKV, -1, -1));
                busesID.push_back(id);
                getline(dataset, line);
                dataset >> line;
            }
        }
        if (line.compare("mpc.gen") == 0) {
        	printf("DGM:: inside loadcase, i = %d, now checking gen data\n", dgm_i);
        	getline(dataset, line);
            dataset >> line;
            int dgm_ii = 1;

            unsigned int busID;  // Gen bus ID
			double P = 0;        // Generated actve power
			double Q = 0;        // Generated reactive power
			double Pmax;         // Max Gen P
			double Pmin;         // Min Gen P
			double Qmax;         // Max Gen Q
			double Qmin;         // Min Gen Q
			double VG;           // Generator voltage

			double tmp_d;  // temp double
			std::string tmp_s;

            while (line.compare("];") != 0) {
            	printf("DGM:: inside loadcase, ii = %d, now checking gen data\n", dgm_ii);
            	cout << "DGM:: inside loadcase, line = " << line << endl;

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

                /////////////
                dataset >> tmp_s;
                cout << "DGM:: gen line dataset #4: " << tmp_s << endl;
                Qmax = ( tmp_s.compare("Inf") == 0 ? DBL_MAX : atof(tmp_s.c_str()) );
                //dataset >> Qmax;
                dataset >> tmp_s;
				cout << "DGM:: gen line dataset #5: " << tmp_s << endl;
				Qmin = ( tmp_s.compare("-Inf") == 0 ? -(DBL_MAX-2) : atof(tmp_s.c_str()) );
                //dataset >> Qmin;
                dataset >> VG;
                buses[busID].VG = VG;
                dataset >> tmp_d;
                dataset >> tmp_d;
                dataset >> Pmax;
                dataset >> Pmin;
                gens.push_back(Generator(busID, P, Q, Pmax, Pmin, Qmax, Qmin));
                getline(dataset, line);
                dataset >> line;

                dgm_ii++;
            }
        }
        if (line.compare("mpc.branch") == 0) {
        	getline(dataset, line);
            dataset >> line;

            printf("DGM:: inside loadcase, i = %d, now checking branch data\n", dgm_i);

            while (line.compare("];") != 0) {
                unsigned int from;  // From Bus
                unsigned int to;    // To Bus
                bool inservice;     // Is the circuit closed or not (open)
                double R;           // Resistance (p.u.)
                double X;           // Reactance (p.u.)
                double B;           // Line Susceptance (p.u.)
                double tap;        // Transormer tap (tap = from.V / to.V)
                double shift;       // defasagem em transformador defasador
//                double Pfrom;       // Active power injected at from
//                double Qfrom;       // Reactive power injected at from
//                double Pto;         // Active power injected at to
//                double Qto;         // Reactive power injected at to

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
        dgm_i++;
    }
    Topology topo(baseMVA, busesID, buses, branches, gens);
    return topo;
}

#endif /* LOADMATPOWERFORMAT_H_ */
