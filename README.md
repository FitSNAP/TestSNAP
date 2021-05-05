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
    Evan Weinberg (NVIDIA)

## Build system
### CMake
The repo has a CMakeLists.txt in the root directory.
It needs two parameters, 1) problem size(-Dref_data=2/8/14) and 2) kokkos installation (-DKokkos_ROOT=/path-to-kokkos-install)
For example to build the benchmark problem size for an NVIDIA GPU using the Kokkos CUDA backend

    cd TestSNAP
    mkdir build_cuda && cd build_cuda
    cmake -Dref_data=14 -DKokkos_ROOT=/path-to-cuda-install ../

### Raw Makefile
There is a basic Makefile provided in the src directory. 
Makefile should be modified to point the KOKKOS_PATH to the Kokkos directory.
Makefile needs two parameters too, 1) problem size(-Dref_data=2/8/14) and 2) the DEVICE type to determing the Kokkos backend
For example to build the benchmark problem size for an NVIDIA GPU using the Kokkos CUDA backend

    cd TestSNAP/src
    make ref_data=14 DEVICE=cuda
<!-----------------END-HEADER------------------------------------->

