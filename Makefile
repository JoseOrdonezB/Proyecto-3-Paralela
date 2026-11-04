all: pgm.o	hough houghConst

hough:	houghBase.cu pgm.o common/pgm.h
	nvcc houghBase.cu pgm.o -o hough

houghConst:	houghConst.cu pgm.o common/pgm.h
	nvcc houghConst.cu pgm.o -o houghConst

pgm.o:	common/pgm.cpp common/pgm.h
	g++ -c common/pgm.cpp -o ./pgm.o

clean:
	rm -f hough houghConst pgm.o
