//#define DEBUG
//#define TEST_MAPoL
#define BASE_INDEX 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <string>
#include <cstdio>
#include <unistd.h>

#include <thrust/host_vector.h>
#include "Eigen/SparseLU"

using namespace std;
using namespace Eigen;

/*
 * Execution parameters
 * Parameters with default values
 */

string	path			= "datasets/case14.m";
char	execucao		= 'P';
uint	N_PARTICULAS	= 1;
uint	N_ITERACOES		= 200;
double	W_MAX			= 0.9;
double	W_MIN			= 0.4;
double	COG				= 2;
double	SOC				= 2;
uint	N_THREADS		= sysconf(_SC_NPROCESSORS_ONLN);
/* End of execution parameters */

double *timeTable;
enum TimeTableIndexes {
	TIME_MAIN = 0,
	TIME_INIT_STRUCT_PSO,
	TIME_ALLOC,
	TIME_PSO,
	TIME_RUNPF,
	TIME_COMPUTEVOLTAGE,
	TIME_MAKEYBUS,
	TIME_NEWTONPF,
	TIME_COMPUTEDIAGIBUS,
	TIME_COMPUTENNZJACOBIANMATRIX,
	TIME_COMPUTEJACOBIANMATRIX,
	TIME_D2H_MEM_COPY,
	TIME_SOLVER_MKL_DSS,
	TIME_H2D_MEM_COPY,
	TIME_UPDATEVOLTAGE,
	TIME_COMPUTE_POWER,
	TIME_CHECKCONVERGENCE,
	TIME_COMPUTELOSS,
	TIME_FREE,
	TIME_TABLE_SIZE
};

string labelNR[] = {
		"|->main",
		"   |->Init Control Struct",
		"   |->Alloc Memory",
		"   |->PSO",
		"      |->RunPF",
		"         |->computeVoltage",
		"         |->MakeYbus",
		"         |->NewtonPF",
		"            |->ComputeDiagIBus",
		"            |->ComputeNnzJacobianMatrix",
		"            |->ComputeJacobianMatrix",
		"            |->MemcpyDeviceToHost",
		"            |->SolverMklDSS",
		"            |->MemcpyHostToDevice",
		"            |->UpdateVoltage",
		"            |->ComputePower",
		"            |->CheckConvergence",
		"         |->computeLoss",
		"   |->Free Memory",
};


// File with results
fstream logFile;

#include <pso/Particula.h>

#define MKL_Complex16 double2
#include "mkl.h"
#include "mkl_dss.h"

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include "util/helper_cuda.h"
#include "util/timer.h"

#include "files/loadMatpowerFormat.h"
#include "powersystem/Topology.h"
#include "powersystem/Bus.h"
#include "powersystem/Branch.h"
#include "powersystem/Generator.h"
#include "powerflow/runpf.h"
#include "util/strings.h"
#include "util/writelog.h"

/*
 * The following help text has been translated from Portuguese to English
 * However, said parameters, in-code, have not been translated yet.
 */
void printHelp(){
	printf(""
			"\n####################################################"
			"\n#####  MAPoL - Minimize Active Power Loss  #########"
			"\n####################################################"
			"\nRuns an active power loss minimization for a given casefile, for a given execution type EXEC-TYPE."
			"\n"
			"\nUsage: MAPoL FILEPATH EXEC-TYPE PART-N ITER-N WMAX WMIN C1 C2"
			"\n\tFILEPATH\tcasefile for powerflow at MATLAB-Matpower format (default: datasets/case14.m)"
			"\n\tEXEC-TYPE\tExecution in a sequential (S); parallel (P) or OpenMP (O) manner (default: S)"
			"\n\tPART-N\t\t(PSO) Particle number"
			"\n\tITER-N\t\t(PSO) Number of iterations"
			"\n\tWMAX\t\t(PSO) Minimum inertia value"
			"\n\tWMIN \t\t(PSO) Maximum inertia value"
			"\n\tC1\t\t(PSO) Cognitive parameter value"
			"\n\tC2\t\t(PSO) Social parameter value"
			"\n\tTHREAD-NUMBER Number of CPU threads"
			"\n\t\tNPART\t- Same number of particles"
			"\n\t\tMAX\t- (default) Maximum number of simultaneous threads on processor"
			"\n\t\t(Enter a specific integer)"
			"\n\tALGORITHM Powerflow solving (NR: Newton-Raphson; FDBX/FDXB: Fast-Decoupled Powerflow; default: NR)"
			"\n\tMETHOD Linear system solution (MKL_DSS: LU decomp. by MKL; SparseLU: LU decomp. by Eigen; default: MKL_DSS)"
			"\n\n");
	exit(1);
}

void readParameters(int argc, char **argv){
	string alg = "NR";
	string linSol = "MKL_DSS";
	if(argc > 1){
		path = string(argv[1]);
		if(path == "--help" || path == "-h"){
			printHelp();
		}
		writeLog(string("arg[1] = ") + path);
	}
	if(argc > 2){
		execucao = argv[2][0];
		writeLog(string("arg[2] = ") + execucao);
	}
	if(argc > 3){
		N_PARTICULAS = atoi(argv[3]);
		writeLog(string("arg[3] = ") + argv[3]);
	}
	if(argc > 4){
		string nthreads(argv[4]);
		if (!nthreads.compare("MAX")){
			N_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
		} else if (!nthreads.compare("NPART")) {
			N_THREADS = N_PARTICULAS;
		} else {
			N_THREADS = atoi(argv[4]);
		}
	}
	if(argc > 5){
		alg = string(argv[5]);
		if(alg == "NR"){
			H_ALG = NR;
		}else if(alg == "FDXB"){
			H_ALG = FDXB;
		}else if(alg == "FDBX"){
			H_ALG = FDBX;
		} else {
			printf("Não foi possivel identificar o algotirmo para calculo de Fluxo de carga: %s\n\n", alg.c_str());
			printHelp();
			exit(1);
		}
	}
	if(argc > 6){
		linSol = string(argv[6]);
		if(linSol == "SparseLU"){
			H_LinearSolver = Eigen_SparseLU;
		}else if(linSol == "MKL_DSS"){
			H_LinearSolver = MKL_DSS;
		}else if(linSol == "cuSolver"){
					H_LinearSolver = cuSolver;
		} else {
			printf("Não foi possivel identificar o metodo de resolução de Sistemas Lineares: %s\n\n", linSol.c_str());
			printHelp();
			exit(1);
		}
	}
	if(argc > 10){
		N_ITERACOES = atoi(argv[7]);
		W_MAX = atof(argv[8]);
		W_MIN = atof(argv[9]);
		COG = atof(argv[10]);
		SOC = atof(argv[11]);
	}

	int found = path.find_last_of("/");
	string fileName;
#ifndef SQL_LOG_FILE
	fileName +=	execucao;
	fileName +=  '_';
	fileName +=  path.substr(found+1);
	fileName +=  '_';
	fileName +=  alg;
	fileName +=  '_';
	fileName +=  linSol;
	fileName +=  '_';
	fileName +=  to_string(N_PARTICULAS);
	fileName +=  ".log";
#else
	fileName = "log.sql";
#endif
	logFile.open(fileName.c_str(), fstream::out | fstream::app);
}

#include <pso/run.h>

int main(int argc, char **argv) {

	readParameters(argc, argv);
	srand(time(0));

	Topology t = loadMatpowerFormat(path);
	printf("DGM:: Finished loading case\n");
	writeLog(string("Loaded test case: ") + path);

	vector<pso::Particula::Estrutura> estrutura;
	for(int i = 0; i < t.buses.size(); i++){
		if(t.buses[i].type == Bus::PV){
			t.buses[i].indiceEstrutura = estrutura.size();
			pso::Particula::Estrutura var = pso::Particula::Estrutura::newAVR(0.9, 1.1, t.buses[i].VG, i);
//			var.id = i;
//			var.tipo = pso::Particula::Estrutura::AVR;
//			var.max = 1.1;
//			var.min = 0.9;
			estrutura.push_back(var);
		}
		if(t.buses[i].Bsh > 0){
			t.buses[i].indiceEstrutura = estrutura.size();
			pso::Particula::Estrutura var = pso::Particula::Estrutura::newShC(0.18, t.buses[i].Bsh, 3, i);
//			var.id = i;
//			var.tipo = pso::Particula::Estrutura::SHC;
//			var.max = 0.18;
//			var.qtd = 3;
			estrutura.push_back(var);
		}
	}
	for(int i = 0; i < t.branches.size(); i++){
		if(t.branches[i].tap > 0){
			t.branches[i].indiceEstrutura = estrutura.size();
			pso::Particula::Estrutura var = pso::Particula::Estrutura::newOLTC(0.9, 1.1, t.branches[i].tap, 20, i);
//			var.id = i;
//			var.tipo = pso::Particula::Estrutura::OLTC;
//			var.max = 1.1;
//			var.min = 0.9;
//			var.qtd = 20;
			estrutura.push_back(var);
		}
	}
	printf("DGM:: Finished creating control particles for PSO\n");

	timeTable = (double*) malloc(sizeof(double) * TIME_TABLE_SIZE);
	for(int i = 0; i < TIME_TABLE_SIZE; i++){
		timeTable[i] = 0;
	}
	double start;

//	clock_t inicio = clock();
	StartTimer();
	vector<double> Mg(estrutura.size());

#ifdef TEST_MAPoL

	switch(execucao){
		case 'S':
		{
			MKL_Set_Num_Threads(1);
			start = GetTimer();
			vector<pso::Particula> enxame(N_PARTICULAS);
			for(int i = 0; i < N_PARTICULAS; i++){
				enxame[i].inicializar(estrutura.size(), &Mg, &estrutura);
				for(int j = 0; j < estrutura.size(); j++){
					switch(estrutura[j].tipo){
					case pso::Particula::Estrutura::SHC:
						enxame[i].X[j] = t.buses[estrutura[j].id].Bsh;
						break;
					case pso::Particula::Estrutura::AVR:
						enxame[i].X[j] = t.buses[estrutura[j].id].VG;
						break;
					case pso::Particula::Estrutura::OLTC:
						enxame[i].X[j] = t.branches[estrutura[j].id].tap;
						break;
					}
				}
			}
			timeTable[TIME_INIT_STRUCT_PSO] += GetTimer() - start;
			start = GetTimer();
			mkl_init(t, N_PARTICULAS, estrutura, H_ALG);
			timeTable[TIME_ALLOC] += GetTimer() - start;
			for(int i = 0; i < N_PARTICULAS; i++){
				start = GetTimer();
				enxame[i].mudarFitness(mkl_runpf(estrutura, enxame[i]));
				timeTable[TIME_RUNPF] += GetTimer() - start;
				//printf("Particula %d = %lf\n", i, enxame[i].fitness());
			}
			start = GetTimer();
			mkl_clean();
			timeTable[TIME_FREE] += GetTimer() - start;
			break;
		}
		case 'O':
		{
			MKL_Set_Num_Threads(32);
			start = GetTimer();
			vector<pso::Particula> enxame(N_PARTICULAS);
			#pragma omp parallel for
			for(int i = 0; i < N_PARTICULAS; i++){
				enxame[i].inicializar(estrutura.size(), &Mg, &estrutura);
				for(int j = 0; j < estrutura.size(); j++){
					switch(estrutura[j].tipo){
					case pso::Particula::Estrutura::SHC:
						enxame[i].X[j] = t.buses[estrutura[j].id].Bsh;
						break;
					case pso::Particula::Estrutura::AVR:
						enxame[i].X[j] = t.buses[estrutura[j].id].VG;
						break;
					case pso::Particula::Estrutura::OLTC:
						enxame[i].X[j] = t.branches[estrutura[j].id].tap;
						break;
					}
				}
			}
			timeTable[TIME_INIT_STRUCT_PSO] += GetTimer() - start;
			start = GetTimer();
			mkl_init(t, N_PARTICULAS, estrutura, H_ALG);
			timeTable[TIME_ALLOC] += GetTimer() - start;
			for(int i = 0; i < N_PARTICULAS; i++){
				start = GetTimer();
				enxame[i].mudarFitness(mkl_runpf(estrutura, enxame[i]));
				timeTable[TIME_RUNPF] += GetTimer() - start;
				//printf("Particula %d = %lf\n", i, enxame[i].fitness());
			}
			start = GetTimer();
			mkl_clean();
			timeTable[TIME_FREE] += GetTimer() - start;
			break;
		}
		case 'P':
		{
			MKL_Set_Num_Threads(32);
			start = GetTimer();
			vector<pso::Particula> enxame(N_PARTICULAS);
			for(int i = 0; i < N_PARTICULAS; i++){
				enxame[i].inicializar(estrutura.size(), &Mg, &estrutura);
				for(int j = 0; j < estrutura.size(); j++){
					switch(estrutura[j].tipo){
					case pso::Particula::Estrutura::SHC:
						enxame[i].X[j] = t.buses[estrutura[j].id].Bsh;
						break;
					case pso::Particula::Estrutura::AVR:
						enxame[i].X[j] = t.buses[estrutura[j].id].VG;
						break;
					case pso::Particula::Estrutura::OLTC:
						enxame[i].X[j] = t.branches[estrutura[j].id].tap;
						break;
					}
				}
			}
			timeTable[TIME_INIT_STRUCT_PSO] += GetTimer() - start;
			start = GetTimer();
			hybrid_init(t, N_PARTICULAS, 32, estrutura, H_ALG);
			timeTable[TIME_ALLOC] += GetTimer() - start;
			start = GetTimer();
			hybrid_runpf(estrutura, enxame);
			timeTable[TIME_RUNPF] += GetTimer() - start;
			for(int i = 0; i < N_PARTICULAS; i++){
				printf("Particula %d = %lf\n", i, enxame[i].fitness());
			}
			start = GetTimer();
			hybrid_free();
			timeTable[TIME_FREE] += GetTimer() - start;
		}
			break;
		default:
		{
			cout << "Unkown EXEC-TYPE. Valid EXEC-TYPEs: S; P" << endl;
			printHelp();
		}
	}

#else
	start = GetTimer();
	switch(execucao){
		case 'S':
		{
			cout << "# of CPU threads: " << N_THREADS << endl;
			MKL_Set_Num_Threads(N_THREADS);
			mkl_init(t, N_PARTICULAS, estrutura, H_ALG);
			break;
		}
		case 'O':
		{
			cout << "# of CPU threads: " << N_THREADS << endl;
			MKL_Set_Num_Threads(N_THREADS);
			mkl_init(t, N_PARTICULAS, estrutura, H_ALG);
			break;
		}
		case 'P':
		{
			cout << "# of CPU threads: " << N_THREADS << endl;
			MKL_Set_Num_Threads(N_THREADS);
			hybrid_init(t, N_PARTICULAS, N_THREADS, estrutura, H_ALG);
			break;
		}
		default:
		{
			cout << "Unkown EXEC-TYPE. Valid EXEC-TYPEs: S; P; O" << endl;
			printHelp();
		}
	}
	timeTable[TIME_ALLOC] += GetTimer() - start;

	writeLog("PSO iniciado");
	vector<pso::Solucao> melhores;
	start = GetTimer();
	pso::run(&melhores, W_MAX, W_MIN, COG, SOC, N_PARTICULAS, N_ITERACOES, estrutura.size(), &estrutura);
	timeTable[TIME_PSO] += GetTimer() - start;
	writeLog((string) melhores[melhores.size() - 1]);

//	double timeTuning = (clock() - inicio) / (double)CLOCKS_PER_SEC;
//	writeLog(string("Tempo de Execucao do Tuning: ") + to_string(timeTuning) + string(" s"));


	start = GetTimer();
	switch(execucao){
			case 'S':
			case 'O':
			{
				mkl_clean();
				break;
			}
			case 'P':
			{
				hybrid_free();
				break;
			}
			default:
			{
				cout << "Unkown EXEC-TYPE. Valid EXEC-TYPEs: S; P" << endl;
				printHelp();
			}
		}
	timeTable[TIME_FREE] += GetTimer() - start;
#endif
    double time = GetTimer();//(clock() - inicio) / (double)CLOCKS_PER_SEC;
    writeLog(string("Execution time: ") + to_string(time) + string(" s"));
	timeTable[TIME_MAIN] += time;


	char buf[1024*1024];
	for(int i = 0; i < TIME_TABLE_SIZE; i++){
		sprintf(buf, "%s(%.3lf s)",labelNR[i].c_str() , timeTable[i]);
		writeLogWithNoDate(string(buf));
	}
#ifdef SQL_LOG_FILE
	stringstream ss;
	ss << "INSERT INTO log (caso_teste,execucao,num_part,pf_alg,ls_alg,num_iter,w_max,w_min,c1,c2,loss,found_iter,found_time,time_main,time_init_struct_pso,time_alloc,time_pso,time_runpf,time_compute_voltage,time_make_y_bus,time_newton_pf,time_compute_diaf_i_bus,time_compute_nnz_jacobian_matrix,time_compute_jacobian_matrix,time_d2h_mem_cpy,time_solver_mkl_dss,time_h2d_mem_cpy,time_update_voltage,time_check_convergence,time_compute_loss,time_free) VALUES ";
	ss << '(';
	int found = path.find_last_of("/");
	ss << '\'' << path.substr(found+1) << "',";
	ss << '\'' << execucao << "',";
	ss << N_PARTICULAS << ',';
	ss << '\'' << H_ALG << "',";
	ss << '\'' << H_LinearSolver << "',";
	ss << N_ITERACOES << ',';
	ss << W_MAX << ',';
	ss << W_MIN << ',';
	ss << COG << ',';
	ss << SOC << ',';
	for (int i = 0; i < TIME_TABLE_SIZE; i++) {
		ss << timeTable[i];
		if (i < TIME_FREE) {
			ss << ',';
		}
	}
	ss << ");";
	ss.getline(buf, 1024*1024);
	writeLog(string(buf), false);
#endif
	logFile.close();
	free(timeTable);
	return 0;
}
