# Esqueleto Overleaf — Modelos de Difusión para IA Generativa

Plantilla LaTeX modular lista para subir a Overleaf y rellenar.

## Cómo usar en Overleaf

1. Comprime esta carpeta en `.zip` y súbela como nuevo proyecto en Overleaf:
   `New Project → Upload Project → seleccionar el .zip`.
2. En Overleaf, asegúrate de que **el compilador esté en `pdfLaTeX`** y el
   **bibliography manager en `Biber`** (Menu → Settings).
3. El fichero raíz es `main.tex`.

## Estructura

```
esqueleto_overleaf/
├── main.tex                          # Documento raíz, includes y setup
├── bibliografia.bib                  # 33 referencias preconfiguradas
├── capitulos/
│   ├── 00_resumen.tex                # Resumen + Abstract (al final)
│   ├── 01_introduccion.tex           # Cap. 1 (al final)
│   ├── 02_estado_arte.tex            # Cap. 2 — la mayoría ya escrito, marcar TODOs
│   ├── 03_desarrollo_software.tex    # Cap. 3 — TODO completo
│   ├── 04_resultados.tex             # Cap. 4 — TODO con estructura
│   └── 05_conclusiones.tex           # Cap. 5 (al final)
├── apendices/
│   └── apendices.tex                 # Apéndices (trasvasar texto existente)
└── figuras/                          # PNG/PDF de gráficas y diagramas
```

## Marcadores `\todo{...}`

Todos los huecos están marcados con `\todo{...}` que sale en rojo en el PDF.
Cuando termines, búscalos con `Ctrl+F` y elimínalos.

Para silenciarlos sin borrarlos, redefine la macro en `main.tex`:
```latex
\renewcommand{\todo}[1]{}
```

## Cosas que ya tienes redactadas y debes pegar

- **§2.2.7 Samplers** → desde `subseccion_samplers.tex` (en la carpeta padre).
- **§2.3.7 Imputación** → desde `subseccion_imputacion.tex`.
- Todo el **Cap. 2 (Estado del Arte)** salvo las dos subsecciones nuevas → de tu PDF actual `IA_Generativa (2).pdf`.
- **Apéndices** → de tu PDF actual.

## Errores técnicos a corregir mientras trasvasas

Marcados también dentro de los `\todo{}` correspondientes:

1. Cosine schedule sin cuadrado en §2.2.5 — debe ser $\cos^2$.
2. NLL sobre train no es válida (§2.4.1).
3. ELBO es desigualdad, no igualdad (§2.4.1).
4. Referencias rotas (2.28), (2.29) en §2.4.4.
5. Renombrar §2.3 a "Generación controlable".
6. Renombrar §5.3 (no es "vs Fokker-Planck", es "EM vs PC vs PF-ODE").
7. Añadir referencia a la U-Net del apéndice desde §2.2.2.

## Compilación local (alternativa a Overleaf)

```bash
pdflatex main
biber main
pdflatex main
pdflatex main
```
