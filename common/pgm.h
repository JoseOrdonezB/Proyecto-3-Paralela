#ifndef PGM_H
#define PGM_H

class PGMImage
{
public:
  int x_dim;
  int y_dim;
  unsigned char *pixels;

  explicit PGMImage (const char *filename);
  ~PGMImage ();

private:
  PGMImage (const PGMImage &);
  PGMImage &operator= (const PGMImage &);
};

#endif
