cmake_minimum_required(VERSION 2.8)
project(libcohog)

include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/..")

#options
option(WITH_CUDA    "Build with CUDA"       ON)
option(WITH_OMP     "Build with OpenMP"     ON)
option(DEBUG        "Build as Debug mode"   OFF)

#HEADERS
set(HEADERS ${HEADERS} libcohog.hpp)
set(HEADERS_SUB ${HEADERS_SUB} CoHOGParams.hpp Detector.hpp HOGDetector.hpp CoHOGDetector.hpp Trainer.hpp Verifier.hpp GroundTruthLoader.hpp)
set(HEADERS_LIBLINEAR ${HEADERS_LIBLINEAR} liblinear/linear.h)

#SOURCES
set(SOURCES ${SOURCES} Detector.cpp HOGDetector.cpp CoHOGDetector.cpp Trainer.cpp Verifier.cpp GroundTruthLoader.cpp)
set(SOURCES ${SOURCES} tinyxml2.cpp)
set(SOURCES ${SOURCES} liblinear/linear.cpp liblinear/tron.cpp liblinear/predict.c liblinear/train.c 
                        liblinear/blas/daxpy.c liblinear/blas/ddot.c liblinear/blas/dnrm2.c liblinear/blas/dscal.c)

set(CUDA_SOURCES "")

#DEBUG
if(DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -pg -DDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

#GoogleTest
add_subdirectory(gtest)

#OpenMP
if(WITH_OMP)
    add_definitions(-DWITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

#CUDA
if(WITH_CUDA)
    set(CUDA_NVCC_FLAGS -arch=sm_20 -O2)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA)
    cuda_add_library(cohog ${SOURCES} ${CUDA_SOURCES})
else()
    add_library(cohog ${SOURCES})
endif()


#Test
add_executable(libcohog_test test_Verifier.cpp)
target_link_libraries(libcohog_test gtest_main)
target_link_libraries(libcohog_test cohog)
target_link_libraries(libcohog_test pthread)
target_link_libraries(libcohog_test opencv_core opencv_highgui opencv_imgproc opencv_objdetect opencv_gpu)

#Install
INSTALL(TARGETS cohog DESTINATION lib)
INSTALL(FILES ${HEADERS_SUB} DESTINATION include/libcohog)
INSTALL(FILES ${HEADERS_LIBLINEAR} DESTINATION include/libcohog/liblinear)
INSTALL(FILES ${HEADERS} DESTINATION include)

