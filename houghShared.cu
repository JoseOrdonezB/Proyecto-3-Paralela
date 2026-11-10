/*
 Proyecto 3 - CUDA: Hough Global + Constante + Compartida (Persona 3)
 Compilar: make houghShared
 Ejecutar: ./houghShared runway.pgm [repeticiones] [k] [salida.png]
 El tiempo medido corresponde exclusivamente al kernel; cudaMemset queda fuera.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <string>
#include "common/pgm.h"
#include "common/lines.h"

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
// Kernel Global + Constante + Compartida. Un hilo por pixel.
// Todos los hilos del bloque deben alcanzar ambas barreras.
__global__ void GPU_HoughTranShared (const unsigned char *pic, int w, int h,
                                      int *acc, float rMax, float rScale)
{
  constexpr int accSize = degreeBins * rBins;
  __shared__ int localAcc[accSize];
  int locID = threadIdx.x;
  int gloID = blockIdx.x * blockDim.x + locID;

  // Inicializacion cooperativa de los 9000 contadores.
  for (int i = locID; i < accSize; i += blockDim.x)
    localAcc[i] = 0;
  __syncthreads();

  // Los hilos fuera de la imagen no votan, pero SI sincronizan.
  if (gloID < w * h && pic[gloID] > 0)
  {
    int xCoord = gloID % w - w / 2;
    int yCoord = h / 2 - gloID / w;
    for (int tIdx = 0; tIdx < degreeBins; ++tIdx)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      if (rIdx >= 0 && rIdx < rBins)
        atomicAdd (&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }
  __syncthreads();

  // Fusion cooperativa del acumulador local con el global.
  // Varios bloques pueden actualizar el mismo bin: se necesita atomicAdd.
  for (int i = locID; i < accSize; i += blockDim.x)
  {
    int votes = localAcc[i];
    if (votes != 0)
      atomicAdd (&acc[i], votes);
  }
}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  if (argc < 2)
    {
      fprintf (stderr, "Uso: %s imagen.pgm [repeticiones] [k] [salida.png]\n", argv[0]);
      return EXIT_FAILURE;
    }
  int reps = (argc >= 3) ? atoi (argv[2]) : 10;
  if (reps < 1) reps = 10;
  double kSigma = (argc >= 4) ? atof (argv[3]) : 2.0;       // umbral = media + k * stdev
  // Guardar siempre las imagenes generadas dentro de resultados/.
  const char *fileName = (argc >= 5) ? argv[4] : "lineas_shared.png";
  const char *baseName = strrchr(fileName, '/');
  baseName = baseName ? baseName + 1 : fileName;
  std::string outputPath = std::string("resultados/") + baseName;
  const char *outPng = outputPath.c_str();
  if (mkdir("resultados", 0755) != 0 && errno != EEXIST) {
    perror("No se pudo crear la carpeta resultados");
    return EXIT_FAILURE;
  }

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
  GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  CUDA_CHECK (cudaGetLastError ());
  CUDA_CHECK (cudaDeviceSynchronize ());

  // Mediciones: se reinicia el acumulador antes de cada corrida
  float *times = (float *) malloc (sizeof (float) * reps);
  double sum = 0.0;
  for (int r = 0; r < reps; r++)
    {
      CUDA_CHECK (cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins));
      CUDA_CHECK (cudaEventRecord (start));
      GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
      CUDA_CHECK (cudaGetLastError ());
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

  // Validacion contra la version secuencial.
  // La CPU calcula r en doble precision (cos/sin devuelven double) y la GPU lo hace
  // en float con multiplicacion-suma fusionada (FMA). Si r cae casi exactamente en
  // el borde entre dos bins, el voto puede caer en el bin vecino (mismo theta,
  // rIdx +-1). Por eso se distingue entre:
  //   - coincidencia exacta,
  //   - diferencias de redondeo: |dif| <= 1 y el total de votos es igual,
  //   - fallo real (cualquier otro caso).
  int mismatches = 0, maxAbsDiff = 0;
  long sumCpu = 0, sumGpu = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    sumCpu += cpuht[i];
    sumGpu += h_hough[i];
    if (cpuht[i] != h_hough[i])
      {
        int d = abs (cpuht[i] - h_hough[i]);
        if (d > maxAbsDiff) maxAbsDiff = d;
        if (mismatches < 20)
          printf ("Calculation mismatch at : %i (r=%d, theta=%d) CPU=%i GPU=%i\n",
                  i, i / degreeBins, i % degreeBins, cpuht[i], h_hough[i]);
        mismatches++;
      }
  }
  bool roundingOnly = (mismatches > 0 && maxAbsDiff <= 1 && sumCpu == sumGpu);
  bool valid = (mismatches == 0) || roundingOnly;
  if (mismatches == 0)
    printf ("Validacion: OK (coincidencia exacta con la CPU)\n");
  else if (roundingOnly)
    printf ("Validacion: OK con tolerancia (%d de %d bins difieren en +-1 por redondeo float/double; votos totales CPU=%ld GPU=%ld)\n",
            mismatches, degreeBins * rBins, sumCpu, sumGpu);
  else
    printf ("Validacion: FALLO (%d diferencias, max |dif| = %d, votos totales CPU=%ld GPU=%ld)\n",
            mismatches, maxAbsDiff, sumCpu, sumGpu);
  printf ("Kernel Global+Constante+Compartida -> promedio: %f ms | desv. est.: %f ms | n = %d\n", mean, stdev, reps);

  // Deteccion de lineas: umbral sobre el acumulador (se usa el resultado de la GPU)
  AccStats st = computeThreshold (h_hough, rBins, degreeBins, kSigma);
  std::vector<HoughLine> lines = detectLines (h_hough, rBins, degreeBins, st.threshold);
  printf ("Acumulador: promedio = %.2f, desv. est. = %.2f -> umbral (k=%.2f) = %.2f\n",
          st.mean, st.stdev, kSigma, st.threshold);
  printf ("Lineas detectadas: %zu\n", lines.size ());
  for (size_t n = 0; n < lines.size (); n++)
    printf ("  linea %zu: theta = %d grados, r = %.1f px, votos = %d\n", n + 1,
            lines[n].tIdx * degreeInc, (lines[n].rIdx + 0.5f) * rScale - rMax, lines[n].votes);
  if (drawLinesPNG (outPng, inImg.pixels, w, h, lines, degreeBins, rMax, rScale, radInc))
    printf ("Imagen guardada en %s\n", outPng);
  else
    fprintf (stderr, "No se pudo escribir %s\n", outPng);

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

  return valid ? 0 : 1;
}
