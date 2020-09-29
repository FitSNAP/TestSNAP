KOKKOS_PATH = $(HOME)/Kokkos/kokkos

ref_data:=2
DEFINE = -DREFDATA_TWOJ=$(ref_data)

LINK = ${CXX}
LINKFLAGS =


	KOKKOS_CXX_STANDARD=c++17
ifeq ($(DEVICE),openmp)
	KOKKOS_DEVICES = "OpenMP"
	CXX = g++
	DEFINE += -fopenmp
	LINKFLAGS += -fopenmp
	EXE = ${EXE_NAME}.openmp
else ifeq ($(DEVICE),cuda)
	KOKKOS_DEVICES = "Cuda"
	CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
	EXE = ${EXE_NAME}.cuda
	KOKKOS_ARCH = "Volta70"
	KOKKOS_CUDA_OPTIONS = "enable_lambda -lineinfo"
else ifeq ($(DEVICE),hip)
        KOKKOS_DEVICES = "HIP"
        CXX = hipcc
        EXE = ${EXE_NAME}.hip
        KOKKOS_ARCH = "Vega906"
else
	KOKKOS_DEVICES = "Serial"
	CXX = g++
	EXE = ${EXE_NAME}.host
endif

CXXFLAGS = -O3 $(DEFINE)

EXE_NAME = "test_snap"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.openmp *.o *.cuda *.hip *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
