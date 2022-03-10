SHELL = /bin/sh

#==========================
# Files
#==========================
EXE = test_snap.exe
SRC = sna.cpp test_snap.cpp memory.cpp

#==========================
# Default compiler is clang
#==========================
COMP ?= clang
COMP_CHOICE ?= $(COMP)

OPENMP := n
OPENMP_TARGET := n

# problem size
ref_data = 2 #2, 8 or 14
DEFINE = -DREFDATA_TWOJ=$(ref_data)

ifeq ($(LMOD_SYSTEM_NAME),summit)
	CUDA_PATH ?= $(OLCF_CUDA_ROOT)
else
	CUDA_PATH?=$(CUDA_ROOT)
endif

ifeq ($(OPENMP),y)
	DEFINE += -DOPENMP
endif

ifeq ($(OPENMP_TARGET),y)
	DEFINE += -DOPENMP_TARGET
endif

CXXFLAGS = -O3

#==========================
# Machine specific info
# compilers and options
#==========================
ifeq ($(COMP),xl)
CC=xlc_r
CXX=xlC_r
CFLAGS += -qsmp=omp:noauto
CFLAGS += -qoffload
CFLAGS += -qtgtarch=sm_70
ifeq ($(LMOD_SYSTEM_NAME),summit)
	CFLAGS += -qarch=pwr9 -qtune=pwr9
endif
	CXXFLAGS += $(CFLAGS) -std=c++11
	CXXFLAGS += $(DEFINE)
endif

ifeq ($(COMP),intel)
	CXX = icc
	CXXFLAGS += -O3 $(DEFINE) -std=c++11 -w
	CXXFLAGS += -qopt-report=5 #Opt Report
	ifeq ($(CRAY_CPU_TARGET),haswell)
		CXXFLAGS += -xcore-avx2
	else ifeq ($(CRAY_CPU_TARGET),mic-knl)
		CXXFLAGS += -xmic-avx512
	endif
ifeq ($(OPENMP),y)
	CXXFLAGS += -qopenmp
endif
endif

ifeq ($(COMP),clang)
CC=clang
CXX=clang++
#CXXFLAGS = -O2 #-ffast-math -ffp-contract=fast -fstrict-aliasing -Werror -Wall -Wno-unused-variable
CXXFLAGS += -fstrict-aliasing #-ffast-math
CXXFLAGS += -Wno-openmp-target -Wall -Wno-unused-variable
CXXFLAGS += $(DEFINE)
CXXFLAGS += -std=c++11
CXXFLAGS += -lm
ifeq ($(OPENMP),y)
	CXXFLAGS += -fopenmp
endif
	ifeq ($(OPENMP_TARGET),y)
		CXXFLAGS += -fopenmp
		CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME} -I${CUDA_HOME}/include -ffp-contract=fast
		ifeq ($(NERSC_HOST),$(filter $(NERSC_HOST),cori perlmutter))
			CXXFLAGS += $(LLVM_CFLAGS)
		endif
	endif
endif

ifeq ($(COMP),nvhpc)
	CXX=nvc++
	CXXFLAGS += -std=c++17
	ifeq ($(OPENMP),y)
		CFLAGS += -mp
	endif
	ifeq ($(OPENMP_TARGET),y)
		CFLAGS += -mp=gpu
		ifeq ($(NERSC_HOST),cori)
			CFLAGS += -gpu=cc70
		endif
		ifeq ($(NERSC_HOST),perlmutter)
			CFLAGS += -gpu=cc80
		endif
	endif
  CXXFLAGS += $(CFLAGS) -std=c++11 $(DEFINE)
endif

ifeq ($(COMP),pgi)
	CXX=pgc++
	CXXFLAGS += -fast -Mlarge_arrays $(DEFINE) -std=c++11
  ifeq ($(OPENMP),y)
    CFLAGS += -mp
	endif
  ifeq ($(OPENMP_TARGET),y)
		CFLAGS += -fast -mp=gpu:cc70
  endif
  CXXFLAGS += $(CFLAGS) -std=c++11 $(DEFINE)
endif

ifeq ($(COMP),gnu)
CC=gcc
CXX=g++
COMP_CHOICE=gcc
CFLAGS += -Ofast -fstrict-aliasing
CFLAGS += -msse -msse2 -mssse3 -msse4.1 -msse4.2 -mfma -mavx -mavx2 -mavx512f -mavx512cd
CFLAGS += -finline-functions -funroll-loops -ftree-vectorize -ftree-loop-optimize -ftree-loop-vectorize -faligned-new
  ifeq ($(OPENMP),y)
    CFLAGS += -fopenmp -pthread
  else ifeq ($(OPENMP_TARGET),y)
    CFLAGS += -fopenmp
    CFLAGS += -foffload=nvptx-none="-Ofast -lm -misa=sm_35"
  else ifeq ($(OPENACC),y)
    CFLAGS += -fopenacc
  endif

  # Compiler flags for different platforms
ifeq ($(LMOD_SYSTEM_NAME),summitdev)
CFLAGS += -mcpu=power8 -mtune=power8
else ifeq ($(LMOD_SYSTEM_NAME),summit)
CFLAGS += -mcpu=power9 -mtune=power9
endif

CXXFLAGS += $(CFLAGS) -std=c++11 $(DEFINE)
FCFLAGS += $(CFLAGS)
endif


#==========================
# Compiler commands
#==========================
CXXLD         = $(CXX)

#==========================
# Make the executable
#==========================
$(EXE): $(SRC) $(INC)
	echo $(SRC)
	$(CXXLD) $(CXXFLAGS) $(SRC) $(PAPI_POST_LINK_OPTS) -o $(EXE)
#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)
