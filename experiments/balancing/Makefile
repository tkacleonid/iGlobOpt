PROGRAM=balancing
BINDIR = build/bin
SRC = src
OBJ = build/obj
INC = include

CC     = nvcc
CLINK  = $(CC)
C_LIB  = -lm
CFLAGS = -std=c++11 -lineinfo -arch=sm_35 -O3 -I${INC} -x cu
CLINKFLAGS= -O3 -arch=sm_35

OBJS = ${OBJ}/kernel.o  ${OBJ}/balancing.o


${BINDIR}/${PROGRAM}: crdir ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${C_LIB}

${OBJ}/kernel.o: ${SRC}/kernel.cpp ${INC}/balancing.hpp
	$(CC) $(CFLAGS) -c ${SRC}/kernel.cpp ${SRC}/balancing.cpp -o ${OBJ}/kernel.o ${OBJ}/balancing.o 
	
${OBJ}/balancing.o: ${SRC}/balancing.cpp ${INC}/balancing.hpp
	$(CC) $(CFLAGS) -c ${SRC}/balancing.cpp -o ${OBJ}/balancing.o 

crdir:
	mkdir -p build
	mkdir -p build/obj
	mkdir -p build/bin

clean:
	rm -f ${OBJ}/*.o
cleanall:
	rm -f ${OBJ}/*.o ${BINDIR}/${PROGRAM}
