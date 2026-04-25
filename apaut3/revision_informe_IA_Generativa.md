# Revisión informe — Modelos de Difusión para IA Generativa

Checklist accionable sobre el estado actual del informe `IA_Generativa.pdf` frente al enunciado `generative_AI_project_AA3_2024_2025.pdf` y al código en `proyecto_AAIII_02_diffusion_models/`.

Leyenda: `[x]` hecho · `[~]` parcial / a pulir · `[ ]` pendiente.

---

## 1. Estado global

- [x] Plantilla TFG EPS UAM aplicada
- [x] Índice, bibliografía con BibTeX (20 referencias, papers canónicos)
- [x] Apéndices A–D (DSM, KL, ELBO, Gaussiana multivariada)
- [~] Estado del Arte avanzado pero con huecos (ver §2)
- [ ] Capítulo 3 Desarrollo — vacío (TODO)
- [ ] Capítulo 4 Resultados — vacío (TODO)
- [ ] Capítulo 5 Conclusiones — vacío (TODO)
- [ ] Resumen / Abstract (inglés + español, 250–500 palabras)
- [ ] Introducción final (cuando esté todo cerrado)

---

## 2. Capítulo 2 — Estado del Arte

### 2.1 Lo que ya está

- [x] Autoencoders y espacios latentes
- [x] Manifold hypothesis
- [x] Movimiento Browniano / Wiener / SDE de Itô
- [x] Integrador de Euler-Maruyama
- [x] U-Net (con figura)
- [x] Proceso forward (inyección de ruido)
- [x] Derivación score matching (Hyvärinen → Vincent → Song) y pérdida multi-escala
- [x] Proceso backward (Anderson 1982) y conexión Langevin
- [x] Variance Exploding (VE)
- [x] Variance Preserving (VP) con schedules lineal y coseno
- [x] Ecuación de Fokker-Planck
- [x] Generación condicionada: classifier guidance y CFG
- [x] SDE backward condicionada (ec. 2.22)

### 2.2 Samplers — ampliar §2.2

- [ ] Sección dedicada a **Predictor-Corrector**: alternancia EM (predictor) + k pasos de Langevin (corrector), criterio de SNR objetivo
- [ ] Sección dedicada a **Probability Flow ODE**: $d\mathbf{x} = [f(\mathbf{x},t) - \tfrac{1}{2}g(t)^2 \nabla \log p_t]\,dt$, determinismo, verosimilitud exacta (base para BPD)
- [ ] Tabla/figura comparativa de los tres samplers (pasos, coste, calidad)

### 2.3 Noise schedules — §2.3 nueva o reubicada

- [ ] Subsección propia con Lineal vs Coseno
- [ ] Curvas de $\bar\alpha_t$ y $\beta(t)$ para cada uno
- [ ] Justificación: por qué coseno aguanta mejor a baja resolución (Nichol & Dhariwal 2021)

### 2.4 Métricas (sección vacía hoy)

- [ ] **BPD**: $\text{BPD} = -\frac{\log p(\mathbf{x})}{D \log 2} + c$ con cambio de variable por Probability Flow ODE (traza del Jacobiano vía Hutchinson)
- [ ] **FID**: $\|\mu_r-\mu_g\|^2 + \mathrm{Tr}(\Sigma_r+\Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ sobre InceptionV3 pool3
- [ ] **IS**: $\exp(\mathbb{E}_{\mathbf{x}}[D_{\mathrm{KL}}(p(y|\mathbf{x})\|p(y))])$ y discusión de limitaciones (insensible a mode collapse intra-clase)

### 2.5 Generación condicionada

- [x] Classifier guidance
- [x] Classifier-free guidance
- [x] SDE backward condicionada
- [ ] **Imputación** (requisito 5.b del enunciado): máscara $M$, sustitución en cada paso $\hat{\mathbf{x}}_t = M\odot \mathbf{x}_t^{\text{known}} + (1-M)\odot \mathbf{x}_t$, conexión con RePaint si aplica

### 2.6 Ventajas / limitaciones por modelo

- [ ] Párrafo o tabla al final de VE y VP con pros/contras (el enunciado lo pide explícitamente)

### 2.7 Double descent (§2.5 actual)

- [ ] Decidir: desarrollar como subsección (definición Belkin/Nakkiran + relación con U-Net) o moverlo a Conclusiones como observación empírica. Tal como está hoy es una nota suelta.

### 2.8 Correcciones puntuales

- [ ] §2.2.3 — discretización de Langevin backward mezcla convenciones; unificar $\Delta t$ vs $\sigma$
- [ ] §2.2.2 ec. (2.1) — índices heterogéneos ($n$ en la suma pero $\mathbf{x}$ sin subíndice)
- [ ] "Introducción al problema" / "Objetivo" / "Estimación de densidad" sin numerar — reconvertir a §2.1 "Introducción y planteamiento"
- [ ] Typo: "Auregressive" → "Autorregresivos"
- [ ] §2.3.5 referencia (2.8) donde debería ser (2.10) — revisar referencias cruzadas
- [ ] Aclarar en la fórmula de Hyvärinen que la suma es sobre dimensiones y $\psi_i$ es la componente i-ésima de $\nabla_x \log p$
- [ ] Añadir figuras del esquema forward (imagen → ruido) y backward (ruido → imagen)

---

## 3. Capítulo 3 — Desarrollo (vacío, alta prioridad)

### 3.1 Análisis de requisitos

- [ ] **Requisitos funcionales** con referencias al Cap. 2 y formulados como testeables
  - [ ] RF sobre procesos de difusión (VE §2.2.4 / VP §2.2.5)
  - [ ] RF sobre samplers (EM, PC, PF-ODE)
  - [ ] RF sobre noise schedules (lineal, coseno)
  - [ ] RF sobre métricas (BPD, FID, IS)
  - [ ] RF sobre generación controlable (class-conditional, imputación)
  - [ ] RF sobre soporte B&N y color
- [ ] **Requisitos no funcionales**
  - [ ] Entorno: Python (ver `.python-version`), PyTorch versión, `uv` + `pyproject.toml` + `uv.lock`
  - [ ] Hardware: GPU/VRAM usadas para entrenamiento e inferencia
  - [ ] Compatibilidad / APIs / formatos de checkpoint (`.pth`)
  - [ ] Rendimiento cuantitativo (p.ej. "generar 64 muestras MNIST en < X s en GPU Y")
  - [ ] Estilo y documentación (black/ruff/docstrings)
  - [ ] Metodología de desarrollo

### 3.2 Casos de uso

- [ ] Caso de uso por notebook: `project_AAIII_teamCode_lastName1_lastName2.ipynb`, `cfg_diffusion_models_SVHN.ipynb`, notebooks de dígitos BW y color
- [ ] Diagrama usuario → librería → salida para cada caso

### 3.3 Planificación

- [ ] Diagrama de **Gantt** con las fases del proyecto (el enunciado da link explícito)

### 3.4 Diseño del paquete

- [ ] Descripción de la estructura `diffusion_lib/{processes, samplers, schedules, metrics}` + `model.py`
- [ ] **Diagrama de clases** (UML sencillo):
  - `BaseProcess` ← `VE`, `VP`
  - `BaseSampler` ← `EulerMaruyama`, `PredictorCorrector`, `ProbabilityFlowODE`, `Imputation`
  - `BaseSchedule` ← `Linear`, `Cosine`
  - `Metrics` → `BPD`, `FID`, `IS`
- [ ] Diagrama de secuencia para training y para sampling

### 3.5 Validación y pruebas

- [ ] Estrategia de tests (unit / integración)
- [ ] Tests automatizados (pytest) — listar qué se cubre

### 3.6 Calidad de software

- [ ] Repositorio en GitHub con README (hoy está vacío)
- [ ] CI con GitHub Actions (lint + tests)
- [ ] Guía de estilo (black / ruff / isort)
- [ ] Generación automática de documentación (Sphinx o mkdocs)

### 3.7 Otras cuestiones

- [ ] Licencia (MIT / Apache-2.0 / GPL — decidir y añadir `LICENSE`)
- [ ] Implicaciones éticas: deepfakes, sesgos de MNIST/SVHN, coste energético

---

## 4. Capítulo 4 — Resultados (vacío)

### 4.1 Galerías comparativas

- [ ] VE vs VP (mismo schedule, mismos pasos)
- [ ] Lineal vs coseno
- [ ] Euler-Maruyama vs Predictor-Corrector vs Probability Flow ODE (mismo seed)
- [ ] Barrido de número de pasos de muestreo
- [ ] B&N (MNIST) y color (SVHN, dígitos color, gatos)

### 4.2 Métricas cuantitativas

- [ ] Tabla FID / IS / BPD por configuración
- [ ] Curvas de entrenamiento (loss vs época)
- [ ] Estudio empírico del double-descent si se mantiene esa línea

### 4.3 Generación controlable

- [ ] Rejilla class-conditional (filas = clase, columnas = muestras)
- [ ] Imputación: original / máscara / reconstrucción

### 4.4 Rendimiento

- [ ] Segundos por imagen y memoria pico por sampler
- [ ] Verificación de los RNF cuantitativos definidos en Cap. 3

---

## 5. Cierre del informe

- [ ] Capítulo 5 Conclusiones (logros, limitaciones, trabajo futuro)
- [ ] Introducción final (se escribe al final según el enunciado)
- [ ] Resumen en español (250–500 palabras)
- [ ] Abstract en inglés (250–500 palabras)
- [ ] Revisión de páginas: objetivo **15–20** (excluyendo portada, índices, bibliografía y apéndices)
- [ ] Revisión final de referencias cruzadas y bibliografía

---

## 6. Orden sugerido de ataque (impacto × esfuerzo)

1. [ ] Cerrar Cap. 2: samplers PC y PF-ODE, §2.4 métricas, imputación, ventajas/limitaciones
2. [ ] Cap. 3 completo (análisis de requisitos + casos de uso + diagrama de clases + Gantt + CI/licencia)
3. [ ] Cap. 4: galerías + tablas de métricas + tiempos
4. [ ] Limpieza de correcciones puntuales del §2.8
5. [ ] Introducción, Resumen y Conclusiones
