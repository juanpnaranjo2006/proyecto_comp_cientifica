"""
Módulo de manejo y procesamiento de imágenes para el proyecto de Computación Científica.
Permite la lectura, exportación, separación/fusión de canales RGB y redimensionamiento
siguiendo las especificaciones del esquema SVD + Interpolación.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image


def recortar_a_cuadrada(imagen: np.ndarray) -> np.ndarray:
    """
    Recorta una imagen a una matriz cuadrada (n, n, 3) tomando la dimensión más pequeña (min(H, W)).
    Si la imagen ya es cuadrada, la retorna sin modificaciones.
    Si no es cuadrada, realiza un recorte centrado y reporta en consola las nuevas dimensiones.

    Parámetros:
    -----------
    imagen : np.ndarray
        Arreglo tridimensional (H, W, 3).

    Retorna:
    --------
    np.ndarray:
        Arreglo tridimensional cuadrado (n, n, 3) donde n = min(H, W).
    """
    if imagen.ndim != 3:
        raise ValueError(f"Se esperaba una imagen tridimensional (H, W, 3), se recibió {imagen.shape}")

    h, w, _ = imagen.shape
    if h == w:
        return imagen

    n = min(h, w)
    inicio_y = (h - n) // 2
    inicio_x = (w - n) // 2
    recortada = imagen[inicio_y : inicio_y + n, inicio_x : inicio_x + n, :]

    print(f"[*] Aviso: Imagen no cuadrada ({w}x{h}). Se recortó centradamente a la dimensión mínima: {n}x{n}")
    return recortada


def leer_imagen(
    ruta: Union[str, Path],
    normalizar: bool = True,
    hacer_cuadrada: bool = True
) -> np.ndarray:
    """
    Carga una imagen desde el disco y la retorna como un arreglo de NumPy en formato RGB.

    Parámetros:
    -----------
    ruta : str | Path
        Ruta del archivo de imagen a leer.
    normalizar : bool, opcional (por defecto True)
        Si es True, convierte los valores de píxeles de [0, 255] a punto flotante en [0.0, 1.0],
        adecuado para operaciones numéricas y factorizaciones SVD como en el artículo.
        Si es False, retorna uint8 en el rango [0, 255].
    hacer_cuadrada : bool, opcional (por defecto True)
        Si es True y la imagen no es cuadrada (H != W), recorta la imagen a min(H, W) x min(H, W)
        y reporta las nuevas dimensiones por consola.

    Retorna:
    --------
    np.ndarray:
        Arreglo tridimensional (H, W, 3) o (n, n, 3) con los valores RGB.
    """
    path = Path(ruta)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen en la ruta especificada: {path.resolve()}")

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)

    if normalizar:
        arr_proc = arr.astype(np.float64) / 255.0
    else:
        arr_proc = arr

    if hacer_cuadrada:
        arr_proc = recortar_a_cuadrada(arr_proc)

    return arr_proc


def separar_canales(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa una imagen RGB (H, W, 3) en sus tres capas o componentes individuales:
    Ar (Rojo), Ag (Verde) y Ab (Azul), según las ecuaciones (2), (3) y (4) del artículo.

    Parámetros:
    -----------
    imagen : np.ndarray
        Arreglo tridimensional (H, W, 3).

    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Tres matrices bidimensionales (H, W) correspondientes a (R, G, B).
    """
    if imagen.ndim != 3 or imagen.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen de dimensiones (H, W, 3), pero se recibió {imagen.shape}")

    r = imagen[:, :, 0]
    g = imagen[:, :, 1]
    b = imagen[:, :, 2]
    return r, g, b


def combinar_canales(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Combina tres matrices bidimensionales de canales (R, G, B) en un único arreglo (H, W, 3).

    Parámetros:
    -----------
    r, g, b : np.ndarray
        Matrices bidimensionales de igual dimensión (H, W).

    Retorna:
    --------
    np.ndarray:
        Arreglo tridimensional (H, W, 3).
    """
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError(
            f"Las dimensiones de los canales deben coincidir. "
            f"R: {r.shape}, G: {g.shape}, B: {b.shape}"
        )

    return np.dstack((r, g, b))


def guardar_imagen(
    ruta: Union[str, Path],
    datos_imagen: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    desnormalizar: bool = True
) -> None:
    """
    Guarda una imagen en disco a partir de un arreglo (H, W, 3) o de una tupla de canales (R, G, B).

    Parámetros:
    -----------
    ruta : str | Path
        Ruta donde se guardará la imagen (e.g. 'images/lena_reconstruida.png').
    datos_imagen : np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]
        Puede ser un arreglo tridimensional (H, W, 3) o una tupla/lista (R, G, B) de matrices (H, W).
    desnormalizar : bool, opcional (por defecto True)
        Si los datos son de punto flotante en [0.0, 1.0], los escala a [0, 255] y castea a uint8.
        Aplica además recorte (clipping) para asegurar que no se exceda el rango válido.
    """
    path = Path(ruta)
    # Crear directorios padres si no existen
    path.parent.mkdir(parents=True, exist_ok=True)

    # Si se pasan canales por separado (R, G, B)
    if isinstance(datos_imagen, (tuple, list)):
        if len(datos_imagen) != 3:
            raise ValueError("Se esperaba una tupla de 3 matrices (R, G, B).")
        arr = combinar_canales(datos_imagen[0], datos_imagen[1], datos_imagen[2])
    elif isinstance(datos_imagen, np.ndarray):
        arr = datos_imagen
    else:
        raise TypeError("datos_imagen debe ser un np.ndarray (H, W, 3) o una tupla (R, G, B).")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen de dimensiones (H, W, 3), se recibió {arr.shape}")

    # Procesar rango de valores
    if desnormalizar or np.issubdtype(arr.dtype, np.floating):
        # Asegurar rango [0, 1] y escalar a [0, 255]
        arr_clipped = np.clip(arr, 0.0, 1.0)
        arr_uint8 = np.round(arr_clipped * 255.0).astype(np.uint8)
    else:
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr_uint8, mode="RGB")
    img.save(path)


def redimensionar_imagen(
    imagen: np.ndarray,
    nuevo_tamano: Tuple[int, int],
    metodo: str = "bicubic"
) -> np.ndarray:
    """
    Redimensiona una imagen a un nuevo tamaño (ancho, alto), útil para generar
    la versión de baja resolución (LR) reducida a la mitad como indica la ecuación (1) del paper.

    Parámetros:
    -----------
    imagen : np.ndarray
        Imagen de entrada en formato (H, W, 3).
    nuevo_tamano : Tuple[int, int]
        Tupla (ancho, alto) / (W, H).
    metodo : str
        Método de remuestreo: 'nearest', 'bilinear', 'bicubic', o 'lanczos'.

    Retorna:
    --------
    np.ndarray:
        Imagen redimensionada con el mismo tipo de datos que la entrada.
    """
    metodos = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    metodo_resample = metodos.get(metodo.lower(), Image.Resampling.BICUBIC)

    es_flotante = np.issubdtype(imagen.dtype, np.floating)
    if es_flotante:
        arr_uint8 = np.round(np.clip(imagen, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr_uint8 = np.clip(imagen, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr_uint8, mode="RGB")
    img_redim = img.resize(nuevo_tamano, resample=metodo_resample)
    resultado = np.array(img_redim)

    if es_flotante:
        return resultado.astype(np.float64) / 255.0
    return resultado
