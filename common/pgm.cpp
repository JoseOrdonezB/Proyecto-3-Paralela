#include "pgm.h"

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
void skipHeaderWhitespace (std::FILE *f)
{
  int c;
  while ((c = std::fgetc (f)) != EOF)
    {
      if (std::isspace (c))
        continue;
      if (c == '#')
        {
          while ((c = std::fgetc (f)) != EOF && c != '\n')
            ;
          continue;
        }
      std::ungetc (c, f);
      return;
    }
}

void fail (const char *message, const char *filename)
{
  std::fprintf (stderr, "PGMImage: %s (%s)\n", message, filename);
  std::exit (EXIT_FAILURE);
}
}

PGMImage::PGMImage (const char *filename) : x_dim (0), y_dim (0), pixels (nullptr)
{
  std::FILE *f = std::fopen (filename, "rb");
  if (f == nullptr)
    fail ("no se pudo abrir el archivo", filename);

  skipHeaderWhitespace (f);
  char magic[3] = {0, 0, 0};
  if (std::fscanf (f, "%2s", magic) != 1 || std::strcmp (magic, "P5") != 0)
    fail ("se esperaba una imagen PGM binaria (P5)", filename);

  skipHeaderWhitespace (f);
  if (std::fscanf (f, "%d", &x_dim) != 1 || x_dim <= 0)
    fail ("ancho invalido", filename);

  skipHeaderWhitespace (f);
  if (std::fscanf (f, "%d", &y_dim) != 1 || y_dim <= 0)
    fail ("alto invalido", filename);

  skipHeaderWhitespace (f);
  int maxval = 0;
  if (std::fscanf (f, "%d", &maxval) != 1 || maxval <= 0 || maxval > 255)
    fail ("maxval invalido (solo se soportan 8 bits)", filename);

  int sep = std::fgetc (f);
  if (sep == '\r')
    {
      int next = std::fgetc (f);
      if (next != '\n')
        std::ungetc (next, f);
    }
  else if (sep == EOF || !std::isspace (sep))
    {
      fail ("falta el separador despues de maxval", filename);
    }

  std::size_t total = static_cast<std::size_t> (x_dim) * static_cast<std::size_t> (y_dim);
  pixels = new unsigned char[total];
  if (pixels == nullptr)
    fail ("no hay memoria para los pixeles", filename);

  if (std::fread (pixels, 1, total, f) != total)
    fail ("archivo truncado o incompleto", filename);

  std::fclose (f);
}

PGMImage::~PGMImage ()
{
  delete[] pixels;
  pixels = nullptr;
}
