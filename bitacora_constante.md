# Bitácora de tiempos: Global + Constante (Persona 2)

**Programa:** `houghConst.cu` (kernel `GPU_HoughTranConst`, 256 hilos por bloque, un hilo por píxel)
**Imagen:** `runway.pgm` (800 x 600)
**GPU:** NVIDIA Tesla T4 (Google Colab), CUDA 13.0, driver 580.82.07
**Método:** CUDA events alrededor del kernel; un lanzamiento de calentamiento sin medir; el acumulador se reinicia antes de cada corrida.
**Comando:** `./houghConst runway.pgm 10 2.0 lineas.png`

Se hicieron dos ejecuciones independientes del programa (cada una con 10 corridas medidas).

| Corrida | Ejecución A (ms) | Ejecución B (ms) |
|---|---|---|
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
| **Promedio** | **1.329245** | **1.541869** |
| **Desv. est.** | **0.008599** | **0.008647** |

## Observaciones

- Dentro de una misma ejecución la variación es muy pequeña (desviación estándar de unos 0.009 ms, menos del 1 %).
- Entre las dos ejecuciones hay una diferencia de aproximadamente 16 % en el promedio (1.329 ms vs 1.542 ms). Es mayor que la variación interna, por lo que se atribuye al estado de la GPU compartida de Colab (relojes, ocupación) entre lanzamientos del proceso. Por eso, para comparar contra la versión Global, ambas deben medirse en la misma sesión de Colab, de forma consecutiva y con el mismo método.
- Resultados de la imagen: acumulador con promedio 1384.06 y desviación estándar 1078.75; con k = 2.0 el umbral es 3541.55 y se detectan 18 líneas; con k = 2.5 el umbral es 4080.93 y se detectan 9.

## Validación contra la versión secuencial

En ambas ejecuciones, 6 de los 9000 bins del acumulador difieren en ±1 respecto a la CPU, en pares con el mismo θ y bins de r vecinos (por ejemplo, θ = 6°, r = 20 y 21), y la suma total de votos coincide. Se debe a redondeo: la CPU calcula `r` en doble precisión y la GPU en `float`, por lo que un voto que cae casi exactamente en el borde entre dos bins puede asignarse al bin vecino. `houghConst.cu` clasifica este caso como "OK con tolerancia" y marca FALLO ante cualquier diferencia mayor a 1 o si el total de votos no coincide.
