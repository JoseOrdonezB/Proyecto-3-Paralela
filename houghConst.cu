/*
 ============================================================================
 Proyecto 3 - CUDA : Transformada lineal de Hough
 Version       : Memoria Global + Memoria Constante   (Persona 2)
 Base          : houghBase.cu (Persona 1, memoria Global + CUDA events)
 To build use  : make houghConst
 To run        : ./houghConst runway.pgm [repeticiones]
 ============================================================================
 Cambios respecto a houghBase.cu:
   - d_Cos y d_Sin ahora viven en memoria Constante (__constant__) con scope
     global, ya no se reservan con cudaMalloc ni se pasan al kernel.
   - El host las llena con cudaMemcpyToSymbol (antes cudaMemcpy).
   - El kernel GPU_HoughTranConst ya no recibe d_Cos / d_Sin como parametros.
   - Se repite la medicion N veces (por defecto 10) para llenar la bitacora.
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Verifica el resultado de cada llamada a la API de CUDA
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf (stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
               cudaGetErrorString (err_));                                    \
      exit (EXIT_FAILURE);                                                    \
    }                                                                         \
  } while (0)

//*****************************************************************
// Tablas de seno y coseno en MEMORIA CONSTANTE (scope global, fuera de main).
// Son de solo lectura para el device: 45 floats cada una = 180 bytes (+180),
// muy por debajo del limite de 64 KB. Todos los hilos de un warp leen el
// mismo tIdx al mismo tiempo, asi que la cache constante hace broadcast.
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************
// Version secuencial (CPU), identica a la base; sirve para validar el kernel
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0)
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // el eje y se invierte
            float theta = 0;
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++;
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// Kernel Global + Constante. Un hilo por pixel.
// Nota: ya NO recibe d_Cos ni d_Sin; los toma de la memoria Constante global.
__global__ void GPU_HoughTranConst (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;      // hilos sobrantes del ultimo bloque

  int xCent = w / 2;
  int yCent = h / 2;

  // gloID es el indice lineal (fila por fila): columna = gloID % w, fila = gloID / w
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // lectura desde memoria Constante (sin calcular sin/cos en el kernel)
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // varios pixeles distintos (hilos distintos) pueden caer en el mismo bin
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  if (argc < 2)
    {
      fprintf (stderr, "Uso: %s imagen.pgm [repeticiones]\n", argv[0]);
      return EXIT_FAILURE;
    }
  int reps = (argc >= 3) ? atoi (argv[2]) : 10;
  if (reps < 1) reps = 10;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // CPU calculation (referencia)
  CPU_HoughTran (inImg.pixels, w, h, &cpuht);

  // Pre-calculo en el host de los valores trigonometricos
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Host -> memoria Constante del device (reemplaza cudaMalloc + cudaMemcpy)
  CUDA_CHECK (cudaMemcpyToSymbol (d_Cos, pcCos, sizeof (float) * degreeBins));
  CUDA_CHECK (cudaMemcpyToSymbol (d_Sin, pcSin, sizeof (float) * degreeBins));

  // Imagen y acumulador en memoria Global
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  CUDA_CHECK (cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h));
  CUDA_CHECK (cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins));
  CUDA_CHECK (cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice));

  // 1 hilo por pixel, bloques de 256 hilos
  int blockNum = (w * h + 255) / 256;

  cudaEvent_t start, stop;
  CUDA_CHECK (cudaEventCreate (&start));
  CUDA_CHECK (cudaEventCreate (&stop));

  // Lanzamiento de calentamiento (no se mide): evita contar la inicializacion del contexto
  CUDA_CHECK (cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins));
  GPU_HoughTranConst <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  CUDA_CHECK (cudaDeviceSynchronize ());

  // Mediciones: se reinicia el acumulador antes de cada corrida
  float *times = (float *) malloc (sizeof (float) * reps);
  double sum = 0.0;
  for (int r = 0; r < reps; r++)
    {
      CUDA_CHECK (cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins));
      CUDA_CHECK (cudaEventRecord (start));
      GPU_HoughTranConst <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
      CUDA_CHECK (cudaEventRecord (stop));
      CUDA_CHECK (cudaEventSynchronize (stop));
      CUDA_CHECK (cudaEventElapsedTime (&times[r], start, stop));
      sum += times[r];
      printf ("Corrida %2d: %f ms\n", r + 1, times[r]);
    }
  double mean = sum / reps;
  double var = 0.0;
  for (int r = 0; r < reps; r++) var += (times[r] - mean) * (times[r] - mean);
  double stdev = (reps > 1) ? sqrt (var / (reps - 1)) : 0.0;

  // Resultado de la ultima corrida -> host
  CUDA_CHECK (cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost));

  // Validacion contra la version secuencial
  int mismatches = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      {
        mismatches++;
        printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
      }
  }
  printf ("Validacion: %s (%d diferencias)\n", mismatches == 0 ? "OK" : "FALLO", mismatches);
  printf ("Kernel Global+Constante -> promedio: %f ms | desv. est.: %f ms | n = %d\n", mean, stdev, reps);

  // Liberacion de recursos (las tablas __constant__ no se liberan: son estaticas del modulo)
  cudaFree (d_in);
  cudaFree (d_hough);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  free (times);
  free (h_hough);
  free (pcCos);
  free (pcSin);
  delete[] cpuht;

  cudaDeviceReset ();

  return mismatches == 0 ? 0 : 1;
}
