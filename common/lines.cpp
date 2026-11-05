// Deteccion de lineas sobre el acumulador de Hough y dibujo en PNG (solo CPU).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "lines.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

AccStats computeThreshold (const int *acc, int rBins, int degreeBins, double k)
{
  long n = (long) rBins * degreeBins;
  double sum = 0.0;
  for (long i = 0; i < n; i++) sum += acc[i];
  double mean = sum / n;
  double var = 0.0;
  for (long i = 0; i < n; i++) var += (acc[i] - mean) * (acc[i] - mean);
  var /= n;
  AccStats s;
  s.mean = mean;
  s.stdev = std::sqrt (var);
  s.threshold = mean + k * s.stdev;
  return s;
}

std::vector<HoughLine> detectLines (const int *acc, int rBins, int degreeBins, double threshold)
{
  std::vector<HoughLine> lines;
  for (int r = 0; r < rBins; r++)
    for (int t = 0; t < degreeBins; t++)
      {
        int v = acc[r * degreeBins + t];
        if (v <= threshold) continue;

        // maximo local 3x3 (theta es circular: 0 y 178 grados son vecinos)
        bool isMax = true;
        for (int dr = -1; dr <= 1 && isMax; dr++)
          for (int dt = -1; dt <= 1; dt++)
            {
              if (dr == 0 && dt == 0) continue;
              int rr = r + dr;
              if (rr < 0 || rr >= rBins) continue;
              int tt = (t + dt + degreeBins) % degreeBins;
              int o = acc[rr * degreeBins + tt];
              // desempate: ante empate gana el de menor indice
              if (o > v || (o == v && (rr * degreeBins + tt) < (r * degreeBins + t)))
                {
                  isMax = false;
                  break;
                }
            }
        if (isMax)
          {
            HoughLine l = {r, t, v};
            lines.push_back (l);
          }
      }
  return lines;
}

static void setRed (unsigned char *rgb, int w, int h, int i, int j)
{
  if (i < 0 || i >= w || j < 0 || j >= h) return;
  unsigned char *p = rgb + 3 * (j * w + i);
  p[0] = 255; p[1] = 0; p[2] = 0;
}

bool drawLinesPNG (const char *outFile, const unsigned char *pic, int w, int h,
                   const std::vector<HoughLine> &lines, int degreeBins,
                   float rMax, float rScale, float radInc)
{
  // imagen original (gris) -> RGB
  unsigned char *rgb = (unsigned char *) malloc ((size_t) w * h * 3);
  if (rgb == NULL) return false;
  for (int p = 0; p < w * h; p++)
    rgb[3 * p] = rgb[3 * p + 1] = rgb[3 * p + 2] = pic[p];

  int xCent = w / 2;
  int yCent = h / 2;

  for (size_t n = 0; n < lines.size (); n++)
    {
      // centro del bin: la transformada usa rIdx = (r + rMax) / rScale (truncado)
      double r = (lines[n].rIdx + 0.5) * rScale - rMax;
      double theta = lines[n].tIdx * (double) radInc;
      double c = std::cos (theta), s = std::sin (theta);

      // Recta en coordenadas centradas: x*cos + y*sin = r, con x = i - xCent, y = yCent - j
      if (std::fabs (s) > std::fabs (c))
        {
          // recta mas bien horizontal: se recorre cada columna i y se despeja y
          for (int i = 0; i < w; i++)
            {
              double x = i - xCent;
              double y = (r - x * c) / s;
              int j = (int) std::lround (yCent - y);
              setRed (rgb, w, h, i, j);
              setRed (rgb, w, h, i, j + 1);   // grosor 2 px
            }
        }
      else
        {
          // recta mas bien vertical: se recorre cada fila j y se despeja x
          for (int j = 0; j < h; j++)
            {
              double y = yCent - j;
              double x = (r - y * s) / c;
              int i = (int) std::lround (x + xCent);
              setRed (rgb, w, h, i, j);
              setRed (rgb, w, h, i + 1, j);   // grosor 2 px
            }
        }
    }

  int ok = stbi_write_png (outFile, w, h, 3, rgb, w * 3);
  free (rgb);
  return ok != 0;
}
