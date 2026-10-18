# Esquema de Ampliación de Imágenes basado en SVD e Interpolación por Splines Cúbicos mediante Números Aleatorios

**Pontificia Universidad Javeriana Cali**  
*Facultad de Ingeniería y Ciencias — Computación Científica*  
*Profesor:* Juan Paz Paternina  
*Autores:* Juan Pablo Naranjo, Mateo Monroy, César A. Zuluaga (2026)

Este repositorio implementa y evalúa el esquema de super-resolución y ampliación de imágenes a color (RGB) propuesto por **Shivani Ranta y Sandipan Gupta (IEEE I4Tech 2022)**: *"Image Enlargement Scheme based on Singular Value Decomposition and Cubic Spline Interpolation Through Random Numbers"*.

---

## 📋 Estructura del Proyecto

```text
proyecto_comp_cientifica/
├── src/
│   ├── image_handler.py     # Lectura, guardado, recorte centrado a cuadrada y manejo de canales RGB
│   ├── matrix_processor.py  # Factorización SVD, expansión de matrices ortogonales y reconstrucción
│   └── main.py              # CLI principal y punto de entrada modular
├── input_images/            # Directorio con imágenes de entrada para pruebas
├── output_images/           # Directorio donde se guardan las imágenes ampliadas, CSV y código LaTeX
├── tester.py                # Script de pruebas por lotes y cálculo de las 6 métricas del paper
├── .gitignore               # Exclusiones de control de versiones
└── README.md                # Documentación de uso y ejecución
```

---

## ⚙️ Requisitos e Instalación

El proyecto requiere **Python 3.8 o superior** y las siguientes librerías científicas estándar:

```bash
pip install numpy scipy pillow
```

---

## 🚀 Guía de Ejecución desde Terminal

### 1. Procesamiento de una Imagen Individual (`src/main.py`)

Para ampliar una sola imagen al doble de su resolución:

```powershell
python src/main.py <ruta_de_la_imagen>
```

#### Ejemplos:
```powershell
# Procesar una imagen específica en input_images
python src/main.py input_images/test_image_01.png

# Procesar con semilla fija para reproducibilidad
python src/main.py input_images/test_image_01.png --seed 42

# Especificar el método (spline por defecto, o literal para estudio académico)
python src/main.py input_images/test_image_01.png --metodo spline
```

#### Parámetros disponibles:
* `ruta_imagen` *(obligatorio)*: Ruta relativa o absoluta del archivo de imagen a procesar.
* `--seed` *(opcional)*: Número entero para fijar la semilla del generador de números aleatorios.
* `--metodo` *(opcional)*:
  * `spline` *(recomendado / por defecto)*: Aplica Splines Cúbicos evaluados en los pasos incrementales aleatorios.
  * `literal`: Aplica la suma directa $U + h$ de la Ec. 11 del paper (para contrastar experimentalmente el error de saturación del artículo).

> [!NOTE]
> Si la imagen de entrada no es cuadrada ($H \neq W$), el sistema realiza automáticamente un recorte centrado al tamaño $\min(H, W) \times \min(H, W)$ para garantizar la compatibilidad con la factorización SVD.

---

### 2. Pruebas por Lotes y Cálculo de Métricas (`tester.py`)

Para evaluar automáticamente todas las imágenes colocadas en la carpeta `input_images/`:

```powershell
python tester.py
```

#### ¿Qué hace este script?
1. **Ampliación $2\times$:** Procesa cada imagen de `input_images/` y guarda el resultado en `output_images/<nombre>_ampliada.<extension>`.
2. **Protocolo Experimental del Paper (Sec. II Ec. 1 y Sec. IV):**
   * Toma la imagen original como referencia (*Ground Truth*, $2n \times 2n$).
   * Reduce la imagen a la mitad ($n \times n$) para simular la versión de baja resolución (LR).
   * Reconstruye a la resolución original ($2n \times 2n$) comparando tres métodos:
     * **Vecino Más Cercano** (*Nearest Neighbor*)
     * **Interpolación Bilineal** (*Bilinear Interpolation*)
     * **Método Propuesto** (*SVD + Cubic Spline con números aleatorios*)
3. **Cálculo de las 6 Métricas de la Página 3 del Paper:**
   * **Entropy** (Ec. 17): Cantidad de información / riqueza de textura.
   * **SNR** (Ec. 18): Relación señal a ruido en decibeles ($dB$).
   * **PSNR** (Ec. 19): Relación señal a ruido pico en decibeles ($dB$).
   * **RMSE** (Ec. 20): Error cuadrático medio respecto a la imagen original.
   * **Mean** (Ec. 21): Brillo medio de la imagen.
   * **SSIM** (Ec. 22): Índice de similitud estructural ($0$ a $1$).
4. **Exportación automática de resultados:**
   * `output_images/metricas_rendimiento.csv`: Archivo en formato CSV con todas las métricas tabuladas.
   * `output_images/tabla_latex.txt`: Código LaTeX listo para copiar y pegar en su documento o informe (incluye tanto la tabla global con todas las imágenes como tablas individuales estilo IEEE Tablas I y II del paper).

---

## 📊 Formatos de Imagen Soportados

El cargador de imágenes soporta:
* **PNG** (`.png`) — *Recomendado (sin pérdida)*
* **BMP** (`.bmp`)
* **TIFF** (`.tif`, `.tiff`)
* **JPEG** (`.jpg`, `.jpeg`)
* **WEBP** (`.webp`)
