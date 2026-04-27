---
title: "Revisión integral del informe — Modelos de Difusión para IA Generativa"
author: "Para Iván Domínguez y Lucas Miranda"
date: "26 de abril de 2026"
geometry: margin=2.2cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

# 1. Estado actual del informe

Versión revisada: `IA_Generativa (2).pdf` (35 páginas, 33 referencias).

**Avances respecto a la primera versión:**

- §2.4 *Métricas* completa (NLL, BPD, FID, IS).
- Apéndices enriquecidos (U-Net, samplers, Wiener, comparativa de métricas).
- Bibliografía pasa de 20 a 33 referencias.
- Cap. 3 con estructura de secciones (aunque todas TODO).
- Cap. 5 con esqueleto de 5 comparativas planteadas.

**Volumen real de contenido:** ~12 páginas con texto efectivo. Objetivo de la rúbrica: 15–20 (excluyendo portada, índices, bibliografía y apéndices).

---

# 2. Lo que falta — Código

Lo bueno: **el código está más completo que el informe**. Implementado y operativo en `proyecto_AAIII_02_diffusion_models/`:

| Componente | Estado |
|---|---|
| Procesos VE y VP | OK (`diffusion_lib/processes/`) |
| Samplers EM, PC, PF-ODE | OK (`diffusion_lib/samplers/`) |
| Imputation sampler | OK (`samplers/imputation.py`) |
| Schedules lineal y coseno | OK (`diffusion_lib/schedules/`) |
| Métricas BPD, FID, IS | OK (`diffusion_lib/metrics/`) |
| Generación condicionada (CFG) | OK (`generacion_condicionada/`) |
| Checkpoints entrenados | OK (3 modelos: MNIST, dígito 6, SVHN-CFG) |

**Pendiente de añadir al código:**

- `README.md` con instalación, uso, ejemplos. Ahora está vacío.
- Fichero `LICENSE` (decidir entre MIT, Apache-2.0 o GPL).
- Carpeta `tests/` con tests unitarios mínimos (al menos uno por módulo).
- GitHub Actions con CI básica (lint con ruff/black + pytest).
- Documentación generada con Sphinx o mkdocs.

---

# 3. Lo que falta — Informe

## 3.1 Estado del Arte (§2)

- **Samplers en cuerpo principal**, no solo en apéndice. Es un requisito explícito (punto 2 del enunciado). Crear §2.2.7 *Integración numérica de la SDE inversa* con versión compacta y referencia al apéndice para detalles. Tienes el `.tex` redactado en `subseccion_samplers.tex`.
- **Imputación**, requisito 5b. Añadir §2.3.7. Tienes el `.tex` en `subseccion_imputacion.tex`.
- **Noise schedules como bloque propio** o nota explícita de que aplican a ambos procesos.
- **§2.5 Double descent**: o se desarrolla con definición Belkin/Nakkiran o se mueve a Conclusiones.
- **Tabla VE vs VP** con ventajas/limitaciones (lo pide el enunciado).

## 3.2 Capítulo 3 — Desarrollo de Software (todo TODO)

- Requisitos funcionales testeables, con referencias al §2.
- Requisitos no funcionales (Python, PyTorch, `uv`, GPU/VRAM, tiempos cuantitativos).
- Casos de uso ligados a tus notebooks.
- Arquitectura del paquete `diffusion_lib/{processes,samplers,schedules,metrics}` + diagrama de clases (`BaseProcess←VE,VP`, `BaseSampler←EM,PC,PF-ODE,Imputation`, `BaseSchedule←Linear,Cosine`).
- Validación, tests, CI/CD, licencia.
- Diagrama de Gantt del proyecto.

## 3.3 Capítulo 4 — *"Desarrollo"* (vacío)

**Eliminar este capítulo.** No encaja en la rúbrica del enunciado.

## 3.4 Capítulo 5 — Resultados (solo títulos)

Estructura recomendada (1 subsección por requisito del enunciado):

- **§5.1 VE vs VP**: galería + FID/IS/BPD + curvas de entrenamiento.
- **§5.2 Lineal vs coseno** (sobre VP): galería + métricas + trayectoria forward generada con tu código.
- **§5.3 EM vs PC vs PF-ODE** *te falta esto*: galería + tabla de tiempos + curva FID vs N + demostración de determinismo del PF-ODE.
- **§5.4 Generación controlable**: 5.4.1 class-conditional con sweep de $w$ + 5.4.2 imputación con varias máscaras *te falta el 5.4.2*.
- **§5.5 Resultados color** (gatos / SVHN): demuestra B&N + color del enunciado.
- **§5.6 Rendimiento** (recomendado): tabla cuantitativa que verifica los RNF de §3.

## 3.5 Capítulo 6 — Conclusiones (vacío)

Logros, limitaciones, trabajo futuro.

## 3.6 Resumen / Abstract / Introducción final

Pendientes (se escriben al final, según el propio enunciado).

---

# 4. Fallos narrativos y estructurales detectados

Estos son los más urgentes de arreglar, antes incluso de añadir contenido:

1. **Dos capítulos llamados "Desarrollo"** (Cap. 3 y Cap. 4). La rúbrica solo prevé uno. **Borrar Cap. 4**.

2. **Samplers solo en apéndice**: es un requisito de primer nivel y deben aparecer en §2 del cuerpo principal.

3. **Referencia rota** en pág. 5: "*Como se desarrollará en la sección 4...*". Cap. 4 está vacío. Reapuntar a Cap. 3.

4. **§2.4.1 NLL — errores técnicos**:
    - "*test (que para este problema pueden ser las mismas que para train)*" — incorrecto: NLL sobre train sobreestima.
    - $\mathbf{y} = \mathbf{x} + u$ con $u\sim U[0,1]^D$ es **dequantization** estándar, hay que nombrarla.
    - $\log p(\mathbf{x}) = \mathbb{E}_q\log p(\mathbf{x}_0|\mathbf{x}_1) - \sum D_{KL}[\dots]$ es la **cota ELBO**, una *desigualdad* $\geq$, no una igualdad.

5. **Numeración rota en §2.4.4**: referencias a las ecuaciones (2.28) y (2.29) que no existen.

6. **§2.3 mal nombrada**: "Modelos de procesos de difusión para generación condicionada" → renombrar a **"Generación controlable"** (matches el enunciado, punto 5).

7. **Título sin sentido en §5.3**: "VE / VP vs Fokker-Planck". Fokker-Planck no es un modelo competidor, es la ecuación que VE/VP satisfacen. Sustituir por "EM vs PC vs PF-ODE".

8. **U-Net en apéndice pero no presentada en cuerpo**: añadir frase en §2.2.2 que apunte al apéndice.

9. **Cosine schedule sin cuadrado en el informe**: en §2.2.5 escribes $f(t) = \cos(\dots)$ pero la canónica (y la de tu código) es $f(t) = \cos^2(\dots)$.

10. **Estructura no cuadra con la rúbrica oficial** (Resumen / Introducción / Estado del arte / Desarrollo SW / Resultados / Conclusiones / Bibliografía / Apéndices = 8 secciones).

---

# 5. Plan de ataque sugerido (por prioridad)

| # | Tarea | Tiempo estimado | Impacto |
|---|---|---|---|
| 1 | Borrar Cap. 4 vacío y renumerar | 5 min | Alto |
| 2 | Arreglar errores técnicos puntuales (sec. 4 de este doc) | 30 min | Alto |
| 3 | Mover samplers compactos al cuerpo + añadir imputación | 1 h | Alto |
| 4 | Llenar Cap. 3 (Desarrollo de Software) | 4–6 h | Muy alto |
| 5 | Generar resultados desde notebooks y rellenar Cap. 5 | 4–8 h | Muy alto |
| 6 | Cerrar Resumen + Introducción + Conclusiones | 2 h | Medio |
| 7 | README + LICENSE + tests mínimos del código | 2 h | Medio |

---

# 6. Recursos adjuntos

- `subseccion_samplers.tex` — §2.2.7 lista para incluir.
- `subseccion_imputacion.tex` — §2.3.7 lista para incluir.
- `revision_informe_IA_Generativa.md` — checklist completo con casillas.
- `esqueleto_overleaf/` — proyecto LaTeX modular para arrancar/refactorizar.
