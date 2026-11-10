# Bitácora de tiempos: memoria Global + Constante

**Programa:** `houghConst.cu` (kernel `GPU_HoughTranConst`; 256 hilos por bloque, un hilo por píxel).  
**Imagen de entrada:** `runway.pgm` (800 × 600 píxeles).  
**Método:** CUDA events alrededor de la ejecución del kernel; calentamiento previo y reinicio del acumulador entre corridas, según la bitácora original.  
**Parámetros de detección:** 10 corridas, `k = 2.0`.

## 1. Mediciones originales — NVIDIA Tesla T4 (Google Colab)

**Entorno documentado:** NVIDIA Tesla T4, CUDA 13.0, controlador 580.82.07.  
**Comando documentado:** `./houghConst runway.pgm 10 2.0 lineas.png`.

Se realizaron dos ejecuciones independientes, cada una con diez corridas medidas:

| Corrida | Ejecución A (ms) | Ejecución B (ms) |
|---:|---:|---:|
| 1 | 1.339392 | 1.545952 |
| 2 | 1.318976 | 1.555136 |
| 3 | 1.327648 | 1.538304 |
| 4 | 1.332992 | 1.534912 |
| 5 | 1.337344 | 1.537440 |
| 6 | 1.314848 | 1.552384 |
| 7 | 1.328064 | 1.539616 |
| 8 | 1.336960 | 1.547200 |
| 9 | 1.320896 | 1.541856 |
| 10 | 1.335328 | 1.525888 |
| **Promedio reportado** | **1.329245** | **1.541869** |
| **Desviación estándar reportada** | **0.008599** | **0.008647** |

**Interpretación:** la variabilidad dentro de cada ejecución fue pequeña. La diferencia entre las medias de las ejecuciones A y B fue apreciable, pero no puede atribuirse con certeza a una causa concreta sin datos adicionales sobre frecuencias, carga y condiciones del dispositivo.

## 2. Medición posterior — NVIDIA GeForce RTX 4070 (WSL2)

**Entorno:** NVIDIA GeForce RTX 4070, Ubuntu 24.04.4 LTS en WSL2.  
**Comando ejecutado:** `./houghConst runway.pgm 10 2.0 resultados/lineas_constante.png`.

| Estadística | Resultado |
|---|---:|
| Cantidad de corridas | 10 |
| **Tiempo promedio del kernel** | **0.877856 ms** |
| Desviación estándar | 0.005083 ms |
| Umbral (k = 2.0) | 3541.55 votos |
| Líneas detectadas | 18 |

**Nota de trazabilidad:** la salida compartida de esta ejecución incluye el resumen estadístico, pero **no las diez duraciones individuales**. No se inventan ni se reconstruyen valores. Para incorporar una tabla detallada de la RTX 4070, se debe conservar la salida completa de las diez corridas.

La imagen se guardó en `resultados/lineas_constante.png`.

## 3. Detección de líneas y validación

Con `k = 2.0`, el acumulador tuvo una media de **1384.06** votos y una desviación estándar de **1078.75**; el umbral fue **3541.55** votos y se detectaron **18 líneas**. La bitácora original de Tesla T4 también registra que con `k = 2.5` el umbral fue **4080.93** y se detectaron **9 líneas**.

La bitácora original documenta que **6 de los 9000 bins** difirieron en una unidad respecto a la versión secuencial y que el total de votos coincidió. El programa considera aceptable esa diferencia bajo una tolerancia de un voto; **no equivale a coincidencia exacta**. Es compatible con diferencias de redondeo al calcular y discretizar distancias, pero determinar la causa exacta requeriría una comprobación específica. La salida de RTX 4070 compartida aquí no incluye las líneas de validación CPU/GPU, por lo que no se afirma una nueva validación independiente para esa corrida.

## 4. Comparación de las tres versiones en RTX 4070

| Versión CUDA | Promedio (ms) | Relación frente a Global |
|---|---:|---:|
| Global | 0.880922 | 1.000× |
| Global + Constante | 0.877856 | 1.003× |
| Global + Constante + Compartida | 0.365782 | 2.408× |

En los resultados disponibles, incorporar memoria Constante redujo el tiempo medio frente a Global aproximadamente **0.35 %**, una diferencia muy pequeña que podría estar dentro de la variación experimental. La versión con memoria Compartida redujo el tiempo medio aproximadamente **58.5 %** frente a Global. Las cifras de Tesla T4 se presentan como evidencia adicional y **no deben mezclarse con las de RTX 4070** para calcular aceleraciones, ya que corresponden a GPU diferentes.

## 5. Implementación de memoria Constante

Los valores trigonométricos precalculados de los 90 ángulos se almacenan en arreglos `__constant__` (`d_Cos` y `d_Sin`) del dispositivo. El programa copia esos valores desde CPU mediante `cudaMemcpyToSymbol`; luego el kernel los consulta sin recibir punteros a dichos arreglos como argumentos. Esto evita recalcular senos y cosenos para cada píxel y permite aprovechar el mecanismo de acceso a memoria Constante. El beneficio de rendimiento depende del patrón de acceso y del hardware, por lo que no necesariamente supone una aceleración importante respecto a otras estrategias de almacenamiento.

## 6. Conclusiones y recomendaciones

1. La implementación Global + Constante dispone de **dos series completas de diez mediciones en Tesla T4** y de **un resumen de diez mediciones en RTX 4070**.
2. En RTX 4070, la diferencia de promedio entre Global y Global + Constante fue de solo **0.003066 ms**; conviene evitar afirmar una mejora concluyente sin repetir las pruebas bajo condiciones controladas.
3. La versión Global + Constante + Compartida obtuvo el menor promedio de las tres en RTX 4070.
4. Se recomienda guardar las diez duraciones individuales de la nueva ejecución en RTX 4070, mantener idénticos los parámetros de entrada y conservar las imágenes y diagramas como evidencia del informe.
