"""
Módulo tester para evaluación y procesamiento por lotes (batch processing).
Calcula las 6 métricas de rendimiento descritas en la Sección III (Página 3)
del artículo de Ranta y Gupta (2022):
  1. Entropy (Ecuación 17)
  2. SNR (Ecuación 18)
  3. PSNR (Ecuación 19)
  4. RMSE (Ecuación 20)
  5. Mean (Ecuación 21)
  6. SSIM (Ecuación 22)

Compara el método propuesto (SVD + Splines Cúbicos con números aleatorios) frente
a los métodos estándar de la literatura (Vecino Más Cercano e Interpolación Bilineal),
exporta las métricas a un archivo CSV y genera el código LaTeX listo para el informe.
"""

import sys
import csv
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import numpy as np

# Asegurar que el directorio src/ esté disponible para importar módulos del proyecto
DIRECTORIO_RAIZ = Path(__file__).resolve().parent
DIRECTORIO_SRC = DIRECTORIO_RAIZ / "src"
if str(DIRECTORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_SRC))

from image_handler import (
    leer_imagen,
    guardar_imagen,
    redimensionar_imagen,
    recortar_a_cuadrada
)
from main import procesar_imagen, ampliar_imagen_svd


EXTENSIONES_SOPORTADAS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ==============================================================================
# 1. IMPLEMENTACIÓN DE LAS 6 MÉTRICAS DE RENDIMIENTO (SECCIÓN III DEL PAPER)
# ==============================================================================

def calcular_entropia(imagen: np.ndarray) -> float:
    """
    Calcula la Entropía de la imagen según la Ecuación (17) del paper:
        Entropy = - sum_{j=1}^n P(j) * log2(P(j))
    donde P(j) representa la probabilidad de la intensidad j en el histograma.
    """
    img_clipped = np.clip(imagen, 0.0, 1.0)
    arr_uint8 = np.round(img_clipped * 255.0).astype(np.uint8)
    conteos, _ = np.histogram(arr_uint8, bins=256, range=(0, 256))
    probabilidades = conteos / conteos.sum()
    probabilidades = probabilidades[probabilidades > 0]
    return float(-np.sum(probabilidades * np.log2(probabilidades)))


def calcular_snr(imagen_original: np.ndarray, imagen_ampliada: np.ndarray) -> float:
    """
    Calcula la relación señal a ruido (SNR) según la Ecuación (18) del paper:
        SNR = 10 * log10 [ sum(I(x,y)^2) / sum((I(x,y) - B(x,y))^2) ]
    """
    i = np.clip(imagen_original, 0.0, 1.0)
    b = np.clip(imagen_ampliada, 0.0, 1.0)
    numerador = np.sum(i ** 2)
    denominador = np.sum((i - b) ** 2)
    if denominador == 0:
        return float("inf")
    return float(10.0 * np.log10(numerador / denominador))


def calcular_psnr(imagen_original: np.ndarray, imagen_ampliada: np.ndarray) -> float:
    """
    Calcula la relación señal a ruido pico (PSNR) según la Ecuación (19) del paper:
        PSNR = 10 * log10 [ max[I(x,y)]^2 / ( (1/mn) * sum((I(x,y) - B(x,y))^2) ) ]
    """
    i = np.clip(imagen_original, 0.0, 1.0)
    b = np.clip(imagen_ampliada, 0.0, 1.0)
    mse = float(np.mean((i - b) ** 2))
    if mse == 0:
        return float("inf")
    max_val = float(np.max(i))
    if max_val == 0:
        max_val = 1.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calcular_rmse(imagen_original: np.ndarray, imagen_ampliada: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio (RMSE) según la Ecuación (20) del paper:
        RMSE = sqrt[ (1/mn) * sum((I(x,y) - B(x,y))^2) ]
    """
    i = np.clip(imagen_original, 0.0, 1.0)
    b = np.clip(imagen_ampliada, 0.0, 1.0)
    return float(np.sqrt(np.mean((i - b) ** 2)))


def calcular_media(imagen: np.ndarray) -> float:
    """
    Calcula el brillo medio (Mean) según la Ecuación (21) del paper:
        Mean = (1 / (m * n)) * sum(B(x, y))
    """
    img_clipped = np.clip(imagen, 0.0, 1.0)
    return float(np.mean(img_clipped))


def calcular_ssim(
    imagen_original: np.ndarray,
    imagen_ampliada: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Calcula el índice de similitud estructural (SSIM) según la Ecuación (22) del paper:
        SSIM(I, B) = [ (2*mu_I*mu_B + c1) * (2*sigma_{I,B} + c2) ] /
                     [ (mu_I^2 + mu_B^2 + c1) * (sigma_I^2 + sigma_B^2 + c2) ]
    con c1 = (k1*L)^2 y c2 = (k2*L)^2, L = 1.0.
    """
    i = np.clip(imagen_original, 0.0, 1.0)
    b = np.clip(imagen_ampliada, 0.0, 1.0)

    l_val = 1.0
    c1 = (k1 * l_val) ** 2
    c2 = (k2 * l_val) ** 2

    mu_i = float(np.mean(i))
    mu_b = float(np.mean(b))
    var_i = float(np.var(i, ddof=1))
    var_b = float(np.var(b, ddof=1))

    # Covarianza muestral sigma_{I, B}
    cov_ib = float(np.cov(i.ravel(), b.ravel())[0, 1])

    numerador = (2.0 * mu_i * mu_b + c1) * (2.0 * cov_ib + c2)
    denominador = (mu_i ** 2 + mu_b ** 2 + c1) * (var_i + var_b + c2)

    if denominador == 0:
        return 1.0
    return float(numerador / denominador)


def evaluar_todas_las_metricas(
    imagen_original: np.ndarray,
    imagen_ampliada: np.ndarray
) -> Dict[str, float]:
    """
    Calcula el conjunto completo de las 6 métricas de la Página 3 del paper.
    """
    return {
        "Entropy": round(calcular_entropia(imagen_ampliada), 4),
        "SNR": round(calcular_snr(imagen_original, imagen_ampliada), 4),
        "PSNR": round(calcular_psnr(imagen_original, imagen_ampliada), 4),
        "RMSE": round(calcular_rmse(imagen_original, imagen_ampliada), 4),
        "Mean": round(calcular_media(imagen_ampliada), 4),
        "SSIM": round(calcular_ssim(imagen_original, imagen_ampliada), 4),
    }


# ==============================================================================
# 2. GENERADORES DE EXPORTACIÓN (CSV Y CÓDIGO LATEX)
# ==============================================================================

def exportar_csv(ruta_csv: Path, filas: List[Dict[str, Any]]) -> None:
    """
    Exporta la lista de resultados a un archivo CSV estructurado.
    """
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["Imagen", "Metodo", "Entropy", "SNR", "PSNR", "RMSE", "Mean", "SSIM"]
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)
    print(f"[+] Archivo CSV exportado con éxito en: {ruta_csv}")


def generar_codigo_latex(filas: List[Dict[str, Any]], ruta_txt: Path) -> None:
    """
    Genera un archivo .txt con el código LaTeX que construye las tablas de resultados
    (tanto agrupada por imágenes como individuales idénticas a las Tablas I y II del paper).
    """
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)

    lineas = [
        "% ======================================================================",
        "% TABLAS LATEX DE MÉTRICAS DE RENDIMIENTO (COMPUTACIÓN CIENTÍFICA)",
        "% Generadas automáticamente por tester.py a partir de las Ecs. (17)-(22)",
        "% ======================================================================",
        "",
        "% --- OPCIÓN 1: Tabla Global Comparativa con todas las imágenes ---",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Comparación de rendimiento de esquemas existentes y propuesto (Ecuaciones 17--22)}",
        "  \\label{tab:comparacion_global}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{|l|l|c|c|c|c|c|c|}",
        "    \\hline",
        "    \\textbf{Imagen} & \\textbf{Método} & \\textbf{Entropy} & \\textbf{SNR (dB)} & \\textbf{PSNR (dB)} & \\textbf{RMSE} & \\textbf{Mean} & \\textbf{SSIM} \\\\",
        "    \\hline"
    ]

    for fila in filas:
        linea = (
            f"    {fila['Imagen']} & {fila['Metodo']} & "
            f"{fila['Entropy']:.4f} & {fila['SNR']:.4f} & {fila['PSNR']:.4f} & "
            f"{fila['RMSE']:.4f} & {fila['Mean']:.4f} & {fila['SSIM']:.4f} \\\\"
        )
        lineas.append(linea)
        if "Proposed" in fila["Metodo"]:
            lineas.append("    \\hline")

    lineas.extend([
        "  \\end{tabular}",
        "  }",
        "\\end{table}",
        "",
        "% --- OPCIÓN 2: Tablas Individuales por Imagen (Estilo Table I & II del Paper Original) ---"
    ])

    # Agrupar por imagen para tablas individuales estilo paper
    imagenes_unicas = sorted(list({f["Imagen"] for f in filas}))
    for img_nombre in imagenes_unicas:
        filas_img = [f for f in filas if f["Imagen"] == img_nombre]
        lineas.extend([
            f"% Tabla para {img_nombre}",
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{Performance comparison of existing and proposed schemes for {img_nombre}}}",
            f"  \\label{{tab:{img_nombre}}}",
            "  \\begin{tabular}{|l|c|c|c|c|c|c|}",
            "    \\hline",
            "    \\textbf{Methods} & \\textbf{Entropy} & \\textbf{SNR} & \\textbf{PSNR} & \\textbf{RMSE} & \\textbf{Mean} & \\textbf{SSIM} \\\\",
            "    \\hline"
        ])
        for f in filas_img:
            lineas.append(
                f"    {f['Metodo']} & {f['Entropy']:.4f} & {f['SNR']:.4f} & "
                f"{f['PSNR']:.4f} & {f['RMSE']:.4f} & {f['Mean']:.4f} & {f['SSIM']:.4f} \\\\"
            )
        lineas.extend([
            "    \\hline",
            "  \\end{tabular}",
            "\\end{table}",
            ""
        ])

    with open(ruta_txt, mode="w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas))

    print(f"[+] Código LaTeX exportado con éxito en: {ruta_txt}")


# ==============================================================================
# 3. PIPELINE DE PRUEBAS Y VALIDACIÓN POR LOTES
# ==============================================================================

def obtener_imagenes_directorio(directorio: Path) -> List[Path]:
    """
    Retorna la lista ordenada de archivos de imagen soportados en un directorio.
    """
    if not directorio.exists():
        return []
    archivos = [
        archivo for archivo in directorio.iterdir()
        if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES_SOPORTADAS
    ]
    return sorted(archivos)


def probar_todas_las_imagenes(
    dir_entrada: Union[str, Path] = "input_images",
    dir_salida: Union[str, Path] = "output_images",
    seed: int = 42,
    metodo: str = "spline"
) -> None:
    """
    Ejecuta el protocolo experimental del paper:
    1. Procesa cada imagen de 'input_images' ampliándola al doble de resolución (2x)
       y guardando el resultado en 'output_images/'.
    2. Ejecuta el benchmark científico (Original vs Redimensionada a la mitad y reconstruida):
       compara el método propuesto (SVD+CS) con Vecino Más Cercano y Bilineal.
    3. Calcula las 6 métricas de la Página 3 (Entropy, SNR, PSNR, RMSE, Mean, SSIM).
    4. Exporta las tablas comparativas a CSV y genera el código LaTeX correspondiente.
    """
    path_entrada = Path(dir_entrada).resolve()
    path_salida = Path(dir_salida).resolve()
    path_salida.mkdir(parents=True, exist_ok=True)

    imagenes = obtener_imagenes_directorio(path_entrada)

    if not imagenes:
        print(f"[!] No se encontraron imágenes válidas en: {path_entrada}")
        print(f"    Extensiones soportadas: {', '.join(sorted(EXTENSIONES_SOPORTADAS))}")
        return

    print("=" * 80)
    print(f"[*] INICIANDO PROTOCOLO DE EVALUACIÓN Y PRUEBAS ({len(imagenes)} imagen(es))")
    print(f"[*] Directorio de entrada : {path_entrada}")
    print(f"[*] Directorio de salida  : {path_salida}")
    print(f"[*] Método seleccionado   : {metodo}")
    print("=" * 80)

    filas_metricas: List[Dict[str, Any]] = []
    tiempo_inicio_total = time.time()

    for idx, img_path in enumerate(imagenes, start=1):
        print(f"\n[{idx}/{len(imagenes)}] Evaluando imagen: {img_path.name}")
        t0 = time.time()

        try:
            # 1. Ampliación directa 2x para uso final
            nombre_salida = f"{img_path.stem}_ampliada{img_path.suffix}"
            ruta_guardado = path_salida / nombre_salida
            img_ampliada, _ = procesar_imagen(
                ruta_entrada=img_path,
                ruta_salida=ruta_guardado,
                seed=seed,
                metodo=metodo
            )
            h_amp, w_amp, _ = img_ampliada.shape
            print(f"    [+] Ampliación 2x completada ({w_amp}x{h_amp}) -> Guardada en: {nombre_salida}")

            # 2. Protocolo de benchmark del Paper (Página 1 Ec. 1 y Página 3 Sec. IV):
            #    I: Imagen original de referencia (2n x 2n)
            #    I_LR: Imagen reducida a la mitad (n x n)
            #    Reconstrucción con: Nearest Neighbor, Bilinear y Proposed Method (SVD+CS)
            img_original = leer_imagen(img_path, normalizar=True, hacer_cuadrada=True)
            w_orig, h_orig, _ = img_original.shape

            # Garantizar dimensión par para reducción y ampliación simétrica exacta
            if w_orig % 2 != 0:
                img_original = img_original[:-1, :-1, :]
                w_orig, h_orig, _ = img_original.shape

            n_lr = w_orig // 2
            # Ecuación (1): Generar Low-Resolution (LR) a la mitad
            img_lr = redimensionar_imagen(img_original, (n_lr, n_lr), metodo="bicubic")

            # Métodos existentes de comparación (Sección IV del paper)
            img_nn = redimensionar_imagen(img_lr, (w_orig, h_orig), metodo="nearest")
            img_bilinear = redimensionar_imagen(img_lr, (w_orig, h_orig), metodo="bilinear")
            img_propuesto = ampliar_imagen_svd(img_lr, seed=seed, metodo=metodo)

            # 3. Cálculo de las 6 métricas de la Página 3
            m_nn = evaluar_todas_las_metricas(img_original, img_nn)
            m_bi = evaluar_todas_las_metricas(img_original, img_bilinear)
            m_prop = evaluar_todas_las_metricas(img_original, img_propuesto)

            # Registrar filas para exportación
            filas_metricas.append({"Imagen": img_path.stem, "Metodo": "Nearest neighbor interpolation", **m_nn})
            filas_metricas.append({"Imagen": img_path.stem, "Metodo": "Bilinear interpolation", **m_bi})
            filas_metricas.append({"Imagen": img_path.stem, "Metodo": "Proposed method (SVD+CS)", **m_prop})

            t_img = time.time() - t0
            print(f"    [+] Métricas calculadas ({t_img:.2f} s):")
            print(f"        - Vecino Cercano : PSNR={m_nn['PSNR']} dB | SSIM={m_nn['SSIM']} | RMSE={m_nn['RMSE']}")
            print(f"        - Bilineal       : PSNR={m_bi['PSNR']} dB | SSIM={m_bi['SSIM']} | RMSE={m_bi['RMSE']}")
            print(f"        - Propuesto      : PSNR={m_prop['PSNR']} dB | SSIM={m_prop['SSIM']} | RMSE={m_prop['RMSE']} | Entropy={m_prop['Entropy']}")

        except Exception as e:
            print(f"    [-] ERROR al procesar {img_path.name}: {e}")

    # 4. Exportar archivos de resultados
    ruta_csv = path_salida / "metricas_rendimiento.csv"
    ruta_latex = path_salida / "tabla_latex.txt"

    if filas_metricas:
        exportar_csv(ruta_csv, filas_metricas)
        generar_codigo_latex(filas_metricas, ruta_latex)

    tiempo_total = time.time() - tiempo_inicio_total
    print("\n" + "=" * 80)
    print(f"[*] Resumen de ejecución:")
    print(f"    - Imágenes procesadas : {len(imagenes)}")
    print(f"    - Archivo CSV         : {ruta_csv}")
    print(f"    - Archivo LaTeX       : {ruta_latex}")
    print(f"    - Tiempo total        : {tiempo_total:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    probar_todas_las_imagenes()
