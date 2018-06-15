#include <iostream>
#include <thrust/host_vector.h>

//#define DEBUG
#define TEST_MAPoL

#define BASE_INDEX 1



#include <fstream>
#include <ctime>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

/*
 * Parâmetros de Execução.
 * Parâmetros com valores default
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
/* Fim Parâmetros de Execução */

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


// Arquivo com resultados
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
#include <util/helper_cuda.h>
#include <util/timer.h>

#include "files/loadMatpowerFormat.h"
#include "powersystem/Topology.h"
#include "powersystem/Bus.h"
#include "powersystem/Branch.h"
#include "powersystem/Generator.h"
#include "powerflow/runpf.h"
#include "util/strings.h"
#include "util/writelog.h"

void printHelp(){
	printf(""
			"\n####################################################"
			"\n#####  MAPoL - Minimize Active Power Loss  #########"
			"\n####################################################"
			"\n"
			"\nUtilização: MAPoL  ARQUIVO EXECUCAO ALGORITMO NºPART NºITER WMAX WMIN C1 C2"
			"\n\tExecuta minimização da perda de potência ativa do caso de teste ARQUIVO de Forma EXECUCAO"
			"\n\tARQUIVO\t caso de teste de Fluxo de Potencia no padrão do pacote do MATLAB MATPOWER. Valor Pardão: datasets/case14.m"
			"\n\tEXECUCAO\t execução de forma Sequencial ou Paralelo ou OpenMP, opções: S ou P ou O: Valor Padrão: S"
			"\n\tNºPART\t número de partículas (PSO)"
			"\n\tNºTHREADS\T número de threads CPU:\n\t\tNPART - Mesmo número de partículas\n\t\tMAX (default) - Máximo número de threads simultâneas no processador\n\t\tInsira um número inteiro específico"
			"\n\tALGORITMO\t Algoritmo para calculo do Fluxo de Carga: NR - Newton-Raphson, FDBX ou FDXB para Fast-Decoupled Power Flow. Valor Padrão: NR"
			"\n\tMETODO\t Metodo para resolução de Sistemas Lineares: MKL_DSS: decom. LU pelo MKL, SparseLU: decom. LU pela Eigen. Valor Padrão: MKL_DSS"
			"\n\tNºITER\t número de iterações (PSO)"
			"\n\tWMAX\t valor máximo de inércia (PSO)"
			"\n\tWMIN\t valor mínimo de inércia (PSO)"
			"\n\tC1\t valor do parâmetro cognitivo (PSO)"
			"\n\tC2\t valor do parâmetro social (PSO)"
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
	srand(time(0));
	readParameters(argc, argv);

	Topology t = loadMatpowerFormat(path);
	writeLog(string("Caso de Teste Carregado: ") + path);

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
			cout << "Comando EXECUCAO desconhecido. Comandos Validos: S e P" << endl;
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
			cout << "Comando EXECUCAO desconhecido. Comandos Validos: S e P" << endl;
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
				cout << "Comando EXECUCAO desconhecido. Comandos Validos: S e P" << endl;
				printHelp();
			}
		}
	timeTable[TIME_FREE] += GetTimer() - start;
#endif
    double time = GetTimer();//(clock() - inicio) / (double)CLOCKS_PER_SEC;
    writeLog(string("Tempo de Execucao: ") + to_string(time) + string(" s"));
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
