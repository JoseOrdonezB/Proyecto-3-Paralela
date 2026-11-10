# Bitácora de rendimiento — Persona 3
## Transformada de Hough: memoria Global + Constante + Compartida

**Equipo de prueba:** NVIDIA GeForce RTX 4070, Ubuntu 24.04.4 LTS en WSL2.  
**Programa:** `houghShared.cu`  
**Entrada:** `runway.pgm`  
**Ejecución:** `./houghShared runway.pgm 10 2.0 lineas_shared.png`  
**Medición:** CUDA events; tiempos del kernel en milisegundos.

## 1. Objetivo
Evaluar la implementación CUDA de la Transformada de Hough que combina memoria Global, Constante y Compartida, verificando sus resultados frente a CPU y registrando al menos diez tiempos de ejecución.

## 2. Implementación
La imagen de entrada y el acumulador final residen en memoria Global. Los arreglos precalculados de senos y cosenos se consultan desde memoria Constante. Cada bloque utiliza un acumulador temporal `localAcc` en memoria Compartida, con `degreeBins * rBins = 90 * 100 = 9000` contadores enteros (aproximadamente 36 000 bytes).

Los hilos del bloque calculan su identificador local `locID` para colaborar en la inicialización de `localAcc`. Después de inicializarlo, una primera barrera `__syncthreads()` asegura que todos los contadores estén listos. Los hilos que representan píxeles activos calculan los votos de Hough y los incrementan mediante `atomicAdd`, evitando pérdidas de actualizaciones simultáneas. Una segunda barrera garantiza que todos los votos locales terminaron antes de combinar los contadores locales con el acumulador Global `acc`; esta combinación también debe ser atómica entre bloques.

Las barreras requieren que todos los hilos del bloque las alcancen: no se debe hacer un `return` anticipado solo porque un hilo quede fuera de los límites de la imagen.

## 3. Mediciones reales

| Corrida | Tiempo (ms) |
|---:|---:|
| 1 | 0.367616 |
| 2 | 0.363520 |
| 3 | 0.363520 |
| 4 | 0.363520 |
| 5 | 0.362496 |
| 6 | 0.363520 |
| 7 | 0.382976 |
| 8 | 0.363520 |
| 9 | 0.363520 |
| 10 | 0.363616 |

**Resumen reportado por el programa:**
- Promedio: **0.365782 ms**
- Desviación estándar: **0.006193 ms**
- Mínimo observado: **0.362496 ms**
- Máximo observado: **0.382976 ms**
- Cantidad de mediciones: **10**

## 4. Validación CPU/GPU
La ejecución reportó **12 456 540 votos totales** tanto para CPU como para GPU. De los **9000 bins**, **8994 coincidieron exactamente**. Los seis bins restantes difirieron en una unidad:

| Bin | CPU | GPU |
|---:|---:|---:|
| 1803 | 1446 | 1445 |
| 1893 | 1506 | 1507 |
| 5931 | 1653 | 1654 |
| 6021 | 1816 | 1815 |
| 6104 | 1642 | 1641 |
| 6194 | 1586 | 1587 |

La validación del programa fue **«OK con tolerancia»**, no coincidencia exacta. Las diferencias son compatibles con efectos de redondeo de punto flotante y discretización, aunque haría falta una prueba específica para confirmar su causa.

## 5. Detección y visualización de líneas
Se aplicó el criterio **umbral = promedio del acumulador + 2 × desviación estándar**:
- Promedio del acumulador: 1384.06 votos
- Desviación estándar: 1078.75 votos
- Umbral calculado: 3541.55 votos
- Líneas detectadas: **18**
- Archivo de imagen: `resultados/lineas_shared.png` (si se utiliza la versión que guarda resultados en esa carpeta).

## 6. Comparación preliminar de rendimiento

| Implementación | Tiempo promedio (ms) |
|---|---:|
| Global | 0.880922 |
| Global + Constante | 1.329245 |
| Global + Constante + Compartida | 0.365782 |

Con estos promedios, la versión Compartida fue **aproximadamente 2.41 veces más rápida** que la versión Global y **3.63 veces más rápida** que la versión Global + Constante. **Advertencia:** la cifra de Constante procede de una bitácora anterior; antes de presentar estas relaciones como comparación experimental definitiva hay que confirmar que las tres pruebas se realizaron con la misma GPU, imagen, parámetros y metodología de medición.

## 7. Diagrama funcional
Generar `resultados/diagrama_memoria_compartida.png` con el script `generar_diagrama_compartida.py` incluido por separado. El diagrama muestra las entradas Global y Constante, la inicialización y votación local, las dos barreras y la reducción atómica al acumulador Global.

## 8. Conclusiones
1. La versión Global + Constante + Compartida compiló y completó las diez ejecuciones en la GPU de prueba.
2. El uso de un acumulador por bloque permite agrupar votos antes de combinarlos en el acumulador Global, reduciendo potencialmente la contención de actualizaciones globales.
3. `atomicAdd` protege las actualizaciones concurrentes; `__syncthreads()` evita que se lean contadores locales sin inicializar o antes de terminar la votación.
4. La validación obtuvo igual número total de votos, con seis discrepancias unitarias que deben reportarse explícitamente.

## 9. Recomendaciones
- Repetir las tres versiones bajo condiciones idénticas y con calentamiento previo.
- Registrar modelo de GPU, tamaño de entrada, tamaño de bloque y versión del compilador.
- Verificar la disponibilidad de memoria Compartida por bloque (9000 enteros ocupan aproximadamente 36 KB).
- Conservar la imagen generada y el diagrama como evidencia del informe.
