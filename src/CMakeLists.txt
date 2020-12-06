cmake_minimum_required(VERSION 3.15)

project(main CXX)
IF(APPLE)
    message("Using MacOS. Using brew g++ compiler version 10.2")
    set(CMAKE_C_COMPILER "/usr/local/Cellar/gcc/10.2.0/bin/gcc-10")
    set(CMAKE_CXX_COMPILER "/usr/local/Cellar/gcc/10.2.0/bin/g++-10")
ENDIF(APPLE)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")

IF (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    message("Compiling in DEBUG mode")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
ELSE()
    message("Compiling in RELEASE mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
ENDIF ()

set(CMAKE_CXX_STANDARD 11)

set(MAIN_SOURCES main.cpp
                 logger.h
                 stopwatch.h
                 base_config.h
                 base_pso.h
                 cpu_config.h
                 serial_pso.h
                 types_cpu_only.h
                 types_general.h)

## MakeFile Option Explanation
## Invoke "CMAKE_CXX_FLAGS_ZZZZ" See Here: http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
## -Wall                         Enable all compiler warnings
## -std=c++11                    Use C++11 version
## -O3                           Maximum compiler optimization
## -DNDEBUG                      Disable assert checking
## -g                            Produce debugging information
## -fno-omit-frame-pointer       Preserve the frame pointer for use with perf
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -std=c++11 -Wall")
#set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -O3 -g -DNDEBUG -std=c++11 -Wall -fno-omit-frame-pointer")

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    message("SUCCESS: CUDA found")

    project(projCuda LANGUAGES CXX CUDA)
    add_library(projCuda STATIC
            cuda_pso.cuh cuda_pso.cu
            cuda_config.cuh cuda_config.cu)
    target_compile_features(projCuda PUBLIC cxx_std_11)
    set_target_properties(projCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_definitions(-DUSE_CUDA)
else(CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA NOT found")
endif(CMAKE_CUDA_COMPILER)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("SUCCESS: OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

    set(MAIN_SOURCES ${MAIN_SOURCES}
                     openmp_pso.h)

    add_definitions(-DUSE_OPENMP)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(STATUS "SUCCESS: OpenMP TEAMS enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcf-protection=none -fno-stack-protector")
        add_definitions(-DUSE_OPENMP_TEAMS)

        set(MAIN_SOURCES ${MAIN_SOURCES}
                         openmp_teams_pso.h)
    else()
        message(WARNING "OpenMP TEAMS not found")
    endif()
else(OPENMP_FOUND)
    message(WARNING "OpenMP NOT found...")
endif(OPENMP_FOUND)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")
find_package( OpenCL )
if (OPENCL_FOUND)
    message("SUCCESS: OpenCL found")
    if(APPLE)
    else(APPLE)
        include_directories( ${OPENCL_INCLUDE_DIR} )
    endif(APPLE)

    add_definitions(-DUSE_OPENCL)
    set(MAIN_SOURCES ${MAIN_SOURCES}
                     opencl_z_pso.h cl_config.h)
else(OPENCL_FOUND)
    message("ERROR: OpenCL NOT found...")
endif(OPENCL_FOUND)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
add_executable(main ${MAIN_SOURCES})
if (CMAKE_CUDA_COMPILER)
    set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(main PUBLIC projCuda)
endif (CMAKE_CUDA_COMPILER)
if (OPENMP_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif (OPENMP_FOUND)
if (OPENCL_FOUND)
    if(APPLE)
        target_link_libraries(main PUBLIC "-framework OpenCL" )
    else(APPLE)
    endif(APPLE)
    target_link_libraries(main PUBLIC ${OPENCL_LIBRARIES})
endif (OPENCL_FOUND)
