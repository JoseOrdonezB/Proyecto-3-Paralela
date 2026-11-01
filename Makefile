all: pgm.o	hough

hough:	houghBase.cu pgm.o common/pgm.h
	nvcc houghBase.cu pgm.o -o hough

pgm.o:	common/pgm.cpp common/pgm.h
	g++ -c common/pgm.cpp -o ./pgm.o

clean:
	rm -f hough pgm.o
