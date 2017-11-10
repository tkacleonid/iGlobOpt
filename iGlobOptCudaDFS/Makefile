PROGRAM=iGlobOptCuda
BINDIR = build/bin
SRC = src
OBJ = build/obj
INC = include

CC     = nvcc
CLINK  = $(CC)
C_LIB  = -lm
CFLAGS = -std=c++14 -lineinfo -arch=sm_30 -O3 -I${INC} -x cu
CLINKFLAGS= -O3 -arch=sm_30

OBJS = ${OBJ}/kernel.o ${OBJ}/CPUGlobalOptimization.o 


${BINDIR}/${PROGRAM}: crdir ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${C_LIB}

${OBJ}/kernel.o: ${SRC}/kernel.cpp ${INC}/interval.h ${INC}/CUDAGlobalOptimization.h ${INC}/CPUGlobalOptimization.h
	$(CC) $(CFLAGS) -c ${SRC}/kernel.cpp -o ${OBJ}/kernel.o 
	
${OBJ}/CPUGlobalOptimization.o: ${SRC}/CPUGlobalOptimization.cpp ${INC}/interval.h ${INC}/CUDAGlobalOptimization.h ${INC}/CPUGlobalOptimization.h
	$(CC) $(CFLAGS) -c ${SRC}/CPUGlobalOptimization.cpp -o ${OBJ}/CPUGlobalOptimization.o

crdir:
	mkdir -p build
	mkdir -p build/obj
	mkdir -p build/bin

clean:
	rm -f ${OBJ}/*.o
cleanall:
	rm -f ${OBJ}/*.o ${BINDIR}/${PROGRAM}
