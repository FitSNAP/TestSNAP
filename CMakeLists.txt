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
FILE(GLOB sources
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
  )


ADD_EXECUTABLE(
  ${MyTarget}
  ${sources}
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
