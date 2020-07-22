<!----------------BEGIN-HEADER------------------------------------>
## TestSNAP

A proxy for the SNAP force calculation in the LAMMPS molecular dynamics package.
When using this software please cite the original SNAP paper:

A. P. Thompson , L.P. Swiler, C.R. Trott, S.M. Foiles, and G.J. Tucker, "Spectral neighbor analysis method for automated generation of quantum-accurate interatomic potentials," J. Comp. Phys., 285 316 (2015).

_Copyright (2019) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### Original author: 
    Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
    http://www.cs.sandia.gov/~athomps 

#### Additional authors (alphabetical): 
    Rahul Gayatri (Lawrence Berkeley National Lab)
    Stan Moore (Sandia National Labs)
    Steve Plimpton (Sandia National Labs)
    Christian Trott (Sandia National Labs)

## Build system
### CMake
The repo has a CMakeLists.txt in the root directory with special flags for OpenMP4.5 specific flags for XL/IBM and LLVM/clang compilers.
It needs two parameters, 1) problem size(-Dref_data=2/8/14) and 2) the OpenMP version to build the parallel code 
For example to build the benchmark problem size for OPENMP and OPENMP4.5 version 

    cd TestSNAP

    mkdir build_OpenMP3 && cd build_OpenMP3
    cmake -Dref_data=14 -DOPENMP=ON ../

    mkdir build_OpenMP4.5 && cd build_OpenMP4.5
    cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -Dref_data=14 -DOPENMP_TARGET=ON ../

### Raw Makefile
There is a basic Makefile provided in the src directory.
Makefile needs two parameters too, 1) problem size(-Dref_data=2/8/14) and 2) the compiler and parallel version
For example to build the benchmark problem size for an NVIDIA GPU using the Kokkos CUDA backend

    cd TestSNAP/src
    make ref_data=14 COMP=clang OPEMP_TARGET=y  
    
<!-----------------END-HEADER------------------------------------->

