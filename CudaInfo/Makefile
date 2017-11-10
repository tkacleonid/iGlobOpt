PROGRAM=CudaInfo
BINDIR = build/bin
SRC = src
OBJ = build/obj
INC = include

CC     = nvcc
CLINK  = $(CC)
C_LIB  = -lm
CFLAGS = -std=c++14 -lineinfo -arch=sm_35 -O3 -I${INC} -x cu
CLINKFLAGS= -O3 -arch=sm_35

OBJS = ${OBJ}/kernel.o 


${BINDIR}/${PROGRAM}: crdir ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${C_LIB}

${OBJ}/kernel.o: ${SRC}/kernel.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/kernel.cpp -o ${OBJ}/kernel.o 
	

crdir:
	mkdir -p build
	mkdir -p build/obj
	mkdir -p build/bin

clean:
	rm -f ${OBJ}/*.o
cleanall:
	rm -f ${OBJ}/*.o ${BINDIR}/${PROGRAM}
