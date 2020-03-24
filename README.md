# Simultaneous Parallel Power Flow Calculations using Hybrid CPU-GPU Approach

The task of online operational planning and control functions in a control center, such as contingency analysis, is computationally demanding and hundreds of simultaneous power flow (PF) calculations should be performed. As practical and cost-effective tools for speedup calculation, this paper presents the implementation of simultaneous parallel PF calculations using GPU. For the practical and fair comparison, the proposed GPU based parallel PF is compared with CPU based PF. The test results using 14, 30, 57, 118, 300, 1354, 2869, and 9241 bus systems for single, 10, 50, 100, 200 simultaneous parallel PF calculations verify the practical speedup of the proposed method.

General Dependencies
* CUDA compatible NVIDIA graphics card.
* Linux OS with nvidia-cuda-toolkit package.
* Eigen (used version is included in this repository).
* Intel Math Kernel Library.

__Note on Windows:__ With some tweakings, this project should work on Windows. The general setup has been proved successfully on CentOS and Ubuntu.

# Setup
This project was created under NVIDIA® Nsight™ Eclipse Edition as its IDE; as such, the gitignore file contains lines for which general Nsight™ settings files and Debug folder will be ignored, as some of those properties will depend on the particular project creation folder and on system's dependent library folders such as Intel Math Kernel Library folders and Eigen.

To configure this project, an user should import this repository, as is, to Nsight, and then configure NVCC compiler and linker options (under your project properties) as explained hereafter.

__Note:__ All these paths may vary depending on your installed version of MKL.

## NVCC Compiler
### Includes
* /opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/
* /opt/intel/mkl/include
* ../src

## NVCC Linker
### Libraries (-l)
* cublas
* cusparse
* cusolver
* mkl_gf_lp64
* mkl_intel_thread
* mkl_core
* iomp5
* pthread
* m

### Library search path (-L)
* /opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/
* /opt/intel/mkl/lib/intel64

### Miscellaneous, -Xlinker options
* -rpath="/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/"
* -rpath="/opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/lib/intel64_lin/"
* -rpath="/opt/intel/mkl/include/"

# Notes

###### *Preprint submitted to Electrical Power and Energy Systems*
