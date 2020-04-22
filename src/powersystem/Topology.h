#ifndef TOPOLOGY_H_
#define TOPOLOGY_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <powersystem/Bus.h>
#include <powersystem/Branch.h>
#include <powersystem/Generator.h>
#include <util/CVector.h>

/*
 * Class in which topology is defined
 *
 * (WARNING: use this class only at host)
 */
struct Topology {

    /*
     * Attributes
     */
    double baseMVA;
    thrust::host_vector<unsigned int> busesID;
    thrust::host_vector<Bus> buses;
    thrust::host_vector<Branch> branches;
    thrust::host_vector<Generator> gens;

    // Bus IDs by type
    unsigned int idSlackBus;
    thrust::host_vector<unsigned int> idPVbuses;
    thrust::host_vector<unsigned int> idPQbuses;

    /*
     * Default constructor
     */
    __host__ Topology() :
            baseMVA(0), busesID(1), buses(1),
            branches(1), gens(1), idSlackBus(0),
            idPVbuses(1), idPQbuses(1) {}

    /*
     * Copy operation constructor
     */
    __host__ Topology(const Topology& other) :
            baseMVA(other.baseMVA), busesID(other.busesID), buses(other.buses),
            branches(other.branches), gens(other.gens), idSlackBus(other.idSlackBus),
            idPVbuses(other.idPVbuses), idPQbuses(other.idPQbuses) {}

    /*
     * Input-parameters constructor
     * Receives the number of elements in each vector
     */
    __host__ Topology(double baseMVA, unsigned int nBuses,
            unsigned int nBranches, unsigned int nGenerators) :
            baseMVA(baseMVA), busesID(nBuses), buses(nBuses), branches(
                    nBranches), gens(nGenerators) {
        thrust::sequence(busesID.begin(), busesID.end());

        for (register unsigned int i = 0; i < buses.size(); ++i) {
            Bus bus = buses[i];
            if (bus.type == Bus::SLACK) {
                idSlackBus = i;
            } else if (bus.type == Bus::PV) {
                idPVbuses.push_back(i);
            } else if (bus.type == Bus::PQ) {
                idPQbuses.push_back(i);
            }
        }
        for (register unsigned int i = 0; i < buses.size(); ++i) {
			if (buses[i].type == Bus::SLACK) {
			} else if (buses[i].type == Bus::PV) {
				for(int j = 0; j < idPVbuses.size(); j++){
					if(idPVbuses[j] == i){
						this->buses[i].indicePVPQ = j;
						break;
					}
				}
			} else if (buses[i].type == Bus::PQ) {
				for(int j = 0; j < idPQbuses.size(); j++){
					if(idPQbuses[j] == i){
						this->buses[i].indicePVPQ = j + idPVbuses.size();
						break;
					}
				}
			}
		}
    }

    /*
     * Constructor which receives thrust::host_vector's and copies its contents
     * into a thrust::device_vector
     */
    __host__ Topology(double baseMVA, thrust::host_vector<unsigned int> busesID,
            thrust::host_vector<Bus> buses,
            thrust::host_vector<Branch> branches,
            thrust::host_vector<Generator> gens) :
            baseMVA(baseMVA), busesID(busesID), buses(buses), branches(
                    branches), gens(gens) {
        for (register unsigned int i = 0; i < buses.size(); ++i) {
            if (buses[i].type == Bus::SLACK) {
                idSlackBus = i;
            } else if (buses[i].type == Bus::PV) {
                idPVbuses.push_back(i);
            } else if (buses[i].type == Bus::PQ) {
                idPQbuses.push_back(i);
            }
        }
        for (register unsigned int i = 0; i < buses.size(); ++i) {
			if (buses[i].type == Bus::SLACK) {
			} else if (buses[i].type == Bus::PV) {
				for(int j = 0; j < idPVbuses.size(); j++){
					if(idPVbuses[j] == i){
						this->buses[i].indicePVPQ = j;
						break;
					}
				}
			} else if (buses[i].type == Bus::PQ) {
				for(int j = 0; j < idPQbuses.size(); j++){
					if(idPQbuses[j] == i){
						this->buses[i].indicePVPQ = j + idPVbuses.size();
						break;
					}
				}
			}
		}
    }

    /*
     * Structure which packets the exported topology vectors
     * for their use at CUDA kernels
     */
    struct Exported {
        double baseMVA;
        CVector<Bus> buses;
        CVector<Branch> branches;
        CVector<Generator> gens;

        // Bus IDs by type
        unsigned int idSlackBus;
        CVector<unsigned int> idPVbuses;
        CVector<unsigned int> idPQbuses;
    };

    /*
     * Exports the vectors comprised by the topology
     * into C language default vectors (by using CVector)
     * so they can be used within CUDA kernels
     */
    __host__ Exported exportTopology() {
        Exported exported;

        exported.baseMVA = baseMVA;
        exported.buses = exportBuses();
        exported.branches = exportBranches();
        exported.gens = exportGenerators();

        // IDs das barras por tipo
        exported.idSlackBus = idSlackBus;
        exported.idPVbuses = exportIdPVbuses();
        exported.idPQbuses = exportIdPQbuses();

        return exported;
    }

    /*
     * Exports the bus vector
     */
    inline __host__ CVector<Bus> exportBuses() {
        CVector<Bus> exported;

        exported.ptr = thrust::raw_pointer_cast(buses.data());
        exported.len = buses.size();

        return exported;
    }

    /*
     * Exports the branches vector
     */
    inline __host__ CVector<Branch> exportBranches() {
        CVector<Branch> exported;

        exported.ptr = thrust::raw_pointer_cast(branches.data());
        exported.len = branches.size();

        return exported;
    }

    /*
     * Exports the generators vector
     */
    inline __host__ CVector<Generator> exportGenerators() {
        CVector<Generator> exported;

        exported.ptr = thrust::raw_pointer_cast(gens.data());
        exported.len = gens.size();

        return exported;
    }

    /*
     * Exports the PV buses list
     */
    inline __host__ CVector<unsigned int> exportIdPVbuses() {
        CVector<unsigned int> exported;

        exported.ptr = thrust::raw_pointer_cast(idPVbuses.data());
        exported.len = idPVbuses.size();

        return exported;
    }

    /*
     * Exports the PQ buses list
     */
    inline __host__ CVector<unsigned int> exportIdPQbuses() {
        CVector<unsigned int> exported;

        exported.ptr = thrust::raw_pointer_cast(idPQbuses.data());
        exported.len = idPQbuses.size();

        return exported;
    }
};

#endif /* TOPOLOGY_H_ */
