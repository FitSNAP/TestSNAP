cmake_minimum_required (VERSION 3.12)
project(matvec
  LANGUAGES C CXX
  )

# don't allow in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(STATUS "Warning! Building from the source directory is not allow")
  message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build from a separate directory")
  message(ERROR "In-source build")
endif()

SET(MyTarget test_snap)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

ADD_EXECUTABLE(
  ${MyTarget}
  ${CMAKE_SOURCE_DIR}/sna.cpp
  ${CMAKE_SOURCE_DIR}/test_snap.cpp
  ${CMAKE_SOURCE_DIR}/memory.cpp
  ${CMAKE_SOURCE_DIR}/test_snap.h
  ${CMAKE_SOURCE_DIR}/sna.h
  ${CMAKE_SOURCE_DIR}/memory.h
  ${CMAKE_SOURCE_DIR}/arrayMDgpu.h
  ${CMAKE_SOURCE_DIR}/arrayMDcpu.h
  )

target_compile_definitions(${MyTarget} PRIVATE REFDATA_TWOJ=${ref_data})
if(${OPENMP})
  find_package(OpenMP REQUIRED)
  target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(${OPENMP_TARGET})
  find_package(CUDA REQUIRED)
  find_package(OpenMP REQUIRED)
  target_compile_definitions(${MyTarget} PRIVATE OPENMP_TARGET)

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  target_compile_options(${MyTarget} PRIVATE -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include -ffp-contract=fast)
  target_link_options(${MyTarget} PRIVATE -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp=libiomp5)
  endif()

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "XL")
  target_compile_options(${MyTarget} PRIVATE -qsmp=omp:noauto -qoffload)
  target_link_options(${MyTarget} PRIVATE -qsmp=omp:noauto -qoffload)
  endif()
  target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
endif()
