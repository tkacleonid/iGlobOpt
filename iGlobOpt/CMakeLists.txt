cmake_minimum_required(VERSION 2.8)

project(iGlobOpt)

set(SRC_ROOT "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_ROOT "${PROJECT_SOURCE_DIR}/include")

set(HEADERS
	${INCLUDE_ROOT}/iGlobOpt.h
	${INCLUDE_ROOT}/testFuncWithLib.h
	${INCLUDE_ROOT}/testFuncWithOwnImpl.h
)

set(SHARED_SOURCES
	${SRC_ROOT}/CPUGlobOptBFS.cpp 
	${SRC_ROOT}/CPUGlobOptBFSWithMmap.cpp 
	${SRC_ROOT}/CPUGlobOptBFSWithOMP.cpp 
)

include_directories(
	${INCLUDE_ROOT} 
	../../OPTITEST/
	../../snowgoose/
	../../NUC/
	../../COMPI/
)
	

add_executable(iGlobOpt ${HEADERS} ${SRC_ROOT}/iGlobOpt.cpp ${SHARED_SOURCES})