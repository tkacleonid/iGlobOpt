cmake_minimum_required(VERSION 2.8)

project(iGlobOpt)

set(SRC_ROOT "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_ROOT "${PROJECT_SOURCE_DIR}/include")

find_package(OpenMP)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS " -std=c++11 ")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(HEADERS
	${INCLUDE_ROOT}/iGlobOpt.h
	${INCLUDE_ROOT}/testFuncWithLib.h
	${INCLUDE_ROOT}/testFuncWithOwnImpl.h
)

set(SHARED_SOURCES
	${SRC_ROOT}/CPUGlobOptBFS.cpp 
	${SRC_ROOT}/CPUGlobOptBFSWithMmap.cpp 
	${SRC_ROOT}/CPUGlobOptBFSWithOMP.cpp 
	${SRC_ROOT}/CPUGlobOptBFSWithMmapAndOMP.cpp
)

include_directories(
	${INCLUDE_ROOT} 
	../../../../OPTITEST/
	../../../../snowgoose/
	../../../../NUC/
	../../../../COMPI/
)
	

add_executable(iGlobOpt ${HEADERS} ${SRC_ROOT}/iGlobOpt.cpp ${SHARED_SOURCES})
