# Bitácora de rendimiento — Persona 1: memoria Global

## 1. Objetivo

Implementar y evaluar la Transformada Lineal de Hough en CUDA utilizando **memoria Global**, medir el tiempo de ejecución del kernel mediante eventos CUDA y contrastar el acumulador obtenido en GPU con la implementación secuencial en CPU.

## 2. Implementación

Se utiliza `houghBase.cu`, con un grid unidimensional y bloques de **256 hilos**. Cada hilo procesa un píxel, identificado por:

```cpp
int gloID = blockIdx.x * blockDim.x + threadIdx.x;
if (gloID >= w * h) return;
```

El número de bloques se calcula redondeando hacia arriba la división del total de píxeles entre 256. La imagen, los arreglos precalculados de senos y cosenos y el acumulador de Hough residen en memoria Global de GPU. Los votos se suman al acumulador mediante `atomicAdd` para evitar condiciones de carrera.

### Coordenadas de los píxeles

```cpp
int xCoord = gloID % w - xCent;
int yCoord = yCent - gloID / w;
```

`gloID % w` obtiene la columna y `gloID / w` obtiene la fila. Al restar el centro horizontal y restar la fila al centro vertical, el origen se traslada al centro de la imagen y se invierte el sentido del eje Y. Para cada píxel activo se evalúa `r = xCoord * cos(theta) + yCoord * sin(theta)` y se incrementa el bin correspondiente del acumulador.

## 3. Metodología de medición

Se realiza una ejecución de calentamiento y luego **10 ejecuciones medidas** con `cudaEventRecord`, `cudaEventSynchronize` y `cudaEventElapsedTime`. Antes de cada medición se reinicia el acumulador con `cudaMemset`, **fuera del intervalo cronometrado**. Por ello, los tiempos reportados corresponden al kernel y no incluyen las transferencias CPU/GPU, la inicialización del acumulador ni la validación en CPU.

### Entorno de pruebas

- Sistema: Ubuntu 24.04.4 LTS sobre WSL2.
- GPU: NVIDIA GeForce RTX 4070 (identificación mostrada parcialmente por `nvidia-smi`).
- Driver NVIDIA reportado: 596.36.
- Compatibilidad CUDA reportada por `nvidia-smi`: 13.2 (**no equivale necesariamente a la versión instalada de `nvcc`**).
- Versión exacta de `nvcc`: no registrada en la salida compartida.
- Imagen de entrada: `runway.pgm`.
- Compilación: `make hough` (exitosa).
- Ejecución: `./hough runway.pgm` (exitosa).

## 4. Bitácora de tiempos reales

| Medición | Tiempo del kernel (ms) |
|---:|---:|
| 1 | 0.880640 |
| 2 | 0.881664 |
| 3 | 0.898816 |
| 4 | 0.879616 |
| 5 | 0.872448 |
| 6 | 0.883712 |
| 7 | 0.883712 |
| 8 | 0.878592 |
| 9 | 0.873472 |
| 10 | 0.876544 |

**Resumen reportado por el programa:**

- Promedio: **0.880922 ms**.
- Mínimo: **0.872448 ms**.
- Máximo: **0.898816 ms**.
- Rango (máximo − mínimo): **0.026368 ms**.

Los tiempos fueron obtenidos en la GPU del equipo y copiados de la salida de ejecución, no son estimaciones.

## 5. Validación contra CPU

El acumulador contiene **9,000 bins**. La comparación con la versión secuencial arrojó **6 bins diferentes**, cada uno con una diferencia absoluta de **1 voto**. Los otros **8,994 bins coinciden exactamente** (99.9333 %).

| Bin | CPU | GPU | Diferencia absoluta |
|---:|---:|---:|---:|
| 1803 | 1446 | 1445 | 1 |
| 1893 | 1506 | 1507 | 1 |
| 5931 | 1653 | 1654 | 1 |
| 6021 | 1816 | 1815 | 1 |
| 6104 | 1642 | 1641 | 1 |
| 6194 | 1586 | 1587 | 1 |

Salida del programa: `Validacion CPU/GPU: CON DIFERENCIAS; bins diferentes=6/9000; diferencia maxima=1`.

**Interpretación:** los resultados son muy próximos, pero **no existe igualdad exacta** entre ambos acumuladores. Una explicación plausible es la diferencia de precisión o redondeo en los cálculos de coma flotante que determinan el bin de cada voto; para confirmarla habría que investigar esos seis casos. La validación puede considerarse satisfactoria **solo si se admite explícitamente una tolerancia de un voto por bin**.

## 6. Análisis y conclusiones

La versión CUDA Global compiló y se ejecutó correctamente en el entorno de pruebas. Las diez mediciones muestran un tiempo medio cercano a **0.881 ms** para el kernel, con un rango reducido de **0.026368 ms**. Estos datos describen el rendimiento de esta ejecución, pero no permiten por sí solos calcular aceleración frente a CPU: para eso se necesita medir también el tiempo secuencial bajo condiciones comparables.

El uso de `atomicAdd` asegura que los votos concurrentes no se pierdan, aunque múltiples hilos que actualizan los mismos bins pueden generar contención en memoria Global. Las pequeñas diferencias frente a CPU deben reportarse y analizarse, no ocultarse.

## 7. Recomendaciones

1. Mantener la misma imagen, GPU y criterio de medición al comparar con las versiones Global + Constante y Global + Constante + Compartida.
2. Registrar la versión exacta de `nvcc` con `nvcc --version` para completar la reproducibilidad.
3. Investigar los seis bins diferentes antes de declarar equivalencia numérica estricta.
4. Distinguir el tiempo exclusivo del kernel del tiempo total de procesamiento al presentar resultados.
5. Conservar la salida de terminal como evidencia de compilación, ejecución y mediciones.

## 8. Comandos de reproducción

```bash
cd ~/Paralela/Proyecto-3-Paralela
make hough
./hough runway.pgm
```
