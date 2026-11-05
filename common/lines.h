#ifndef LINES_H
#define LINES_H

#include <vector>

// Una linea detectada en el espacio de Hough
struct HoughLine
{
  int rIdx;    // indice del bin de r
  int tIdx;    // indice del bin de theta
  int votes;   // peso (votos) del bin
};

// Estadisticas del acumulador, usadas para definir el umbral
struct AccStats
{
  double mean;
  double stdev;
  double threshold;  // mean + k * stdev
};

// Calcula media y desviacion estandar de TODO el acumulador (rBins * degreeBins)
// y el umbral = media + k * desviacion estandar.
AccStats computeThreshold (const int *acc, int rBins, int degreeBins, double k);

// Devuelve los bins con peso > threshold que ademas son maximo local en su
// vecindad 3x3 (con theta circular), para no dibujar decenas de lineas casi iguales.
std::vector<HoughLine> detectLines (const int *acc, int rBins, int degreeBins, double threshold);

// Dibuja las lineas en rojo sobre la imagen en escala de grises y la guarda como PNG.
// pic: pixeles w*h (1 byte). rMax/rScale/radInc: mismos parametros usados en la transformada.
// Devuelve true si se escribio el archivo.
bool drawLinesPNG (const char *outFile, const unsigned char *pic, int w, int h,
                   const std::vector<HoughLine> &lines, int degreeBins,
                   float rMax, float rScale, float radInc);

#endif
