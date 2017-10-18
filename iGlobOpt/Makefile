CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		iGlobOpt.o

LIBS =

TARGET =	iGlobOpt

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
