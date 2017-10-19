PROGRAM = iGlobOpt
BUILDDIR = build
BINDIR = build/bin
SRC = src
OBJ = build/obj
INC = include
ROOT = ./

CXX = g++
CXXLINK = $(CXX)
CXXLIB = -lm
CXXFLAGS =	-std=c++14 -Wall -03  -I${INC) ${EXTRA_INC} -fopenMP
CXXLINKFLAGS = 03 - fopenmp

OBJS = ${OBJ}/iGlobOpt.o.o \
	${OBJ}/CPUGlobOptBFS.o \

LIBS = $(CXXLIB)

TARGET =	$(PROGRAM)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

	
	${BINDIR}/${PROGRAM}: crdir ${OBJS}
	${CXXLINK} ${CXXLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${CXXLIB}

crdir:
	mkdir -p obj
	mkdir -p bin
	
${OBJ}/main.o: ${SRC}/main.cpp ${INC}/interval.h ${INC}/test_func_with_lib.h 
	$(CC) $(CFLAGS) -c ${SRC}/main.cpp -o ${OBJ}/main.o
${OBJ}/CPU_GO_DFS.o: ${SRC}/CPU_GO_DFS.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/CPU_GO_DFS.cpp -o ${OBJ}/CPU_GO_DFS.o
${OBJ}/CPU_GO_FULL_SPLIT.o: ${SRC}/CPU_GO_FULL_SPLIT.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/CPU_GO_FULL_SPLIT.cpp -o ${OBJ}/CPU_GO_FULL_SPLIT.o
${OBJ}/CPU_GO_FULL_SPLIT_00.o: ${SRC}/CPU_GO_FULL_SPLIT_00.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/CPU_GO_FULL_SPLIT_00.cpp -o ${OBJ}/CPU_GO_FULL_SPLIT_00.o
${OBJ}/CPU_GO_FULL_SPLIT_OMP.o: ${SRC}/CPU_GO_FULL_SPLIT_OMP.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/CPU_GO_FULL_SPLIT_OMP.cpp -o ${OBJ}/CPU_GO_FULL_SPLIT_OMP.o
${OBJ}/boxes_sort.o: ${SRC}/boxes_sort.cpp 
	$(CC) $(CFLAGS) -c ${SRC}/boxes_sort.cpp -o ${OBJ}/boxes_sort.o

all: ${BINDIR}/${PROGRAM}

clean:
	rm -rf $(BUILDDIR)
cleanall:
	rm -rf $(BUILDDIR)
	
	
	
	
	all:	$(BINDIR)/$(TARGET)

clean:
	rm -rf $(BUILDDIR)
	
