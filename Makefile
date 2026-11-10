all: pgm.o	hough houghConst houghShared

hough:	houghBase.cu pgm.o common/pgm.h
	nvcc houghBase.cu pgm.o -o hough

houghConst:	houghConst.cu pgm.o lines.o common/pgm.h common/lines.h
	nvcc houghConst.cu pgm.o lines.o -o houghConst

houghShared: houghShared.cu pgm.o lines.o common/pgm.h common/lines.h
	nvcc houghShared.cu pgm.o lines.o -o houghShared

pgm.o:	common/pgm.cpp common/pgm.h
	g++ -c common/pgm.cpp -o ./pgm.o

lines.o:	common/lines.cpp common/lines.h common/stb_image_write.h
	g++ -O2 -c common/lines.cpp -o ./lines.o

clean:
	rm -f hough houghConst houghShared pgm.o lines.o lineas.png
