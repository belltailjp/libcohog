cmake_minimum_required(VERSION 2.8)
project(libcohog)

include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/..")

#options
option(WITH_CUDA    "Build with CUDA"       ON)
option(WITH_OMP     "Build with OpenMP"     ON)
option(DEBUG        "Build as Debug mode"   OFF)

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
    cuda_add_library(cohog Detector.cpp HOGDetector.cpp CoHOGDetector.cpp Trainer.cpp Verifier.cpp)    #cuがあればここに
else()
    add_library(cohog Detector.cpp HOGDetector.cpp CoHOGDetector.cpp Trainer.cpp Verifier.cpp)
endif()

#DEMO
add_executable(demo Demo.cpp)
target_link_libraries(demo cohog)
target_link_libraries(demo linear)
target_link_libraries(demo glog)
target_link_libraries(demo opencv_core opencv_highgui opencv_imgproc opencv_objdetect opencv_gpu)


#Test
#if(WITH_CUDA)
#    cuda_add_executable(libcohog_test libcohog_test.cpp libcohog_test.cu)
#else()
#    add_executable(libcohog_test libcohog_test.cpp)
#endif()
#target_link_libraries(libcohog_test gtest_main)
#target_link_libraries(libcohog_test cohog)
#target_link_libraries(libcohog_test pthread)

#Install
INSTALL(TARGETS cohog DESTINATION lib)
INSTALL(FILES CoHOGParams.hpp Detector.hpp HOGDetector.hpp CoHOGDetector.hpp Trainer.hpp Verifier.hpp DESTINATION include/libcohog)
INSTALL(FILES libcohog.hpp DESTINATION include)

