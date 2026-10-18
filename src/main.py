"""
Módulo principal y punto de entrada para el esquema de ampliación de imágenes
basado en Descomposición en Valores Singulares (SVD) e interpolación con
números aleatorios (Ranta & Gupta, 2022).
"""

import sys
import argparse
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np

# Asegurar que el directorio de este módulo esté en el path de búsqueda de Python
sys.path.append(str(Path(__file__).resolve().parent))

from image_handler import (
    leer_imagen,
    guardar_imagen,
    separar_canales,
    combinar_canales
)
from matrix_processor import (
    SVD,
    generate_random_incremental_vector,
    expand_feature_pair,
    reconstruct
)


def ampliar_canal_svd(
    canal: np.ndarray,
    seed: Optional[int] = None,
    metodo: str = "spline"
) -> np.ndarray:
    """
    Aplica el algoritmo de ampliación basado en SVD e incrementos aleatorios a un único canal 2D.

    Parámetros:
    -----------
    canal : np.ndarray
        Matriz bidimensional (n, n) del canal de color (R, G o B).
    seed : int, opcional
        Semilla para el generador de números aleatorios para reproducibilidad.
    metodo : str, opcional ('spline' o 'literal')
        Método de expansión ('spline' por defecto).

    Retorna:
    --------
    np.ndarray:
        Canal ampliado al doble de resolución (2n, 2n).
    """
    n_filas, n_columnas = canal.shape
    if n_filas != n_columnas:
        raise ValueError(f"El canal debe ser una matriz cuadrada (n, n), se recibió: {canal.shape}")

    # 1. Descomposición en Valores Singulares: A = U * Sigma * V^T
    u, sigma, vt = SVD(canal, retornar_sigma_diagonal=True)

    # 2. Generación del vector incremental aleatorio h = (r_1, r_2, ..., r_n)
    h = generate_random_incremental_vector(n_columnas, seed=seed)

    # 3. Expansión de características: U -> U_new (2n x n) y V -> (V_new)^T (n x 2n)
    u_new, vt_new = expand_feature_pair(u, vt, h, metodo=metodo)

    # 4. Reconstrucción: B = U_new @ Sigma @ (V_new)^T -> (2n x 2n)
    canal_ampliado = reconstruct(sigma, u_new, vt_new, recortar_rango=True)

    return canal_ampliado


def ampliar_imagen_svd(
    imagen: np.ndarray,
    seed: Optional[int] = None,
    metodo: str = "spline"
) -> np.ndarray:
    """
    Aplica el algoritmo de ampliación a una imagen RGB completa (H, W, 3).
    Separa los tres canales (R, G, B), procesa cada uno de forma independiente mediante SVD
    y vuelve a combinarlos para producir la imagen de alta resolución (2H, 2W, 3).

    Parámetros:
    -----------
    imagen : np.ndarray
        Arreglo tridimensional (H, W, 3) con valores en [0.0, 1.0].
    seed : int, opcional
        Semilla para los vectores de números aleatorios.
    metodo : str, opcional ('spline' o 'literal')

    Retorna:
    --------
    np.ndarray:
        Imagen ampliada de dimensiones (2H, 2W, 3).
    """
    # Separar canales RGB según ecuaciones (2), (3) y (4)
    r, g, b = separar_canales(imagen)

    # Procesar cada componente de forma independiente
    seed_r = seed if seed is None else seed
    seed_g = None if seed is None else seed + 1
    seed_b = None if seed is None else seed + 2

    r_ampliado = ampliar_canal_svd(r, seed=seed_r, metodo=metodo)
    g_ampliado = ampliar_canal_svd(g, seed=seed_g, metodo=metodo)
    b_ampliado = ampliar_canal_svd(b, seed=seed_b, metodo=metodo)

    # Recombinar canales en la imagen final SVD true color
    return combinar_canales(r_ampliado, g_ampliado, b_ampliado)


def procesar_imagen(
    ruta_entrada: Union[str, Path],
    ruta_salida: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    metodo: str = "spline"
) -> Tuple[np.ndarray, Path]:
    """
    Función de lógica desacoplada para leer, procesar mediante SVD + Random y guardar la imagen.
    Diseñada para ser importada directamente por scripts de prueba, benchmarks o testers.

    Parámetros:
    -----------
    ruta_entrada : str | Path
        Ruta del archivo de imagen a procesar.
    ruta_salida : str | Path, opcional
        Ruta donde guardar la imagen procesada. Si es None, se genera automáticamente
        como '<nombre_original>_ampliada.<extension>' en la misma carpeta.
    seed : int, opcional
        Semilla para reproducibilidad.
    metodo : str, opcional ('spline' o 'literal')

    Retorna:
    --------
    Tuple[np.ndarray, Path]:
        - Arreglo tridimensional de la imagen ampliada resultante.
        - Ruta (Path) donde fue guardada la imagen.
    """
    path_entrada = Path(ruta_entrada)
    if not path_entrada.exists():
        raise FileNotFoundError(f"El archivo especificado no existe: {path_entrada.resolve()}")

    # 1. Cargar imagen normalizada en [0.0, 1.0]
    imagen = leer_imagen(path_entrada, normalizar=True)

    # 2. Ejecutar la ampliación mediante el esquema SVD
    imagen_ampliada = ampliar_imagen_svd(imagen, seed=seed, metodo=metodo)

    # 3. Determinar la ruta de salida
    if ruta_salida is None:
        path_salida = path_entrada.parent / f"{path_entrada.stem}_ampliada{path_entrada.suffix}"
    else:
        path_salida = Path(ruta_salida)

    # 4. Guardar la imagen reconstruida
    guardar_imagen(path_salida, imagen_ampliada, desnormalizar=True)

    return imagen_ampliada, path_salida


def main() -> None:
    """
    Punto de entrada de línea de comandos. Lee un argumento obligatorio
    con la ruta relativa de la imagen y ejecuta el procesamiento.
    """
    parser = argparse.ArgumentParser(
        description="Ampliación de imágenes mediante SVD e interpolación con números aleatorios."
    )
    parser.add_argument(
        "ruta_imagen",
        type=str,
        help="Ruta relativa o absoluta de la imagen a procesar."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para los vectores de números aleatorios (para reproducibilidad)."
    )
    parser.add_argument(
        "--metodo",
        type=str,
        choices=["spline", "literal"],
        default="spline",
        help="Método de interpolación: 'spline' (Cubic Splines asistido por random, recomendado) o 'literal' (Ec. 11 directa)."
    )

    args = parser.parse_args()

    try:
        print(f"[*] Iniciando procesamiento para: {args.ruta_imagen} (Método: {args.metodo})")
        img_ampliada, ruta_salida = procesar_imagen(args.ruta_imagen, seed=args.seed, metodo=args.metodo)
        h_amp, w_amp, _ = img_ampliada.shape
        w_orig, h_orig = w_amp // 2, h_amp // 2

        print(f"[+] Proceso completado exitosamente.")
        print(f"[+] Dimensiones ampliadas finales: {w_amp}x{h_amp} (a partir de dimensión base {w_orig}x{h_orig})")
        print(f"[+] Imagen guardada en: {ruta_salida}")

    except Exception as e:
        print(f"[-] Error durante el procesamiento: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
