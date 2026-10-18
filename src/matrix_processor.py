"""
Módulo de procesamiento matricial para el esquema SVD + Interpolación.
Implementa la factorización SVD, generación de vectores de incremento aleatorio,
expansión de matrices de características ortogonales (ecuaciones 11-13 del paper)
y reconstrucción del canal en alta resolución (ecuaciones 14-16).
"""

from typing import Tuple, Union, Optional
import numpy as np


def SVD(matriz: np.ndarray, retornar_sigma_diagonal: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la Descomposición en Valores Singulares (SVD) de una matriz bidimensional
    correspondiente a un canal de color (Ar, Ag o Ab), de acuerdo con las ecuaciones (5), (6) y (7):
        A = U @ Sigma @ V^T

    Parámetros:
    -----------
    matriz : np.ndarray
        Matriz bidimensional (n, n) con los valores del canal.
    retornar_sigma_diagonal : bool, opcional (por defecto True)
        Si es True, retorna Sigma como matriz diagonal 2D (n, n) tal como en el paper [Sr, Sg, Sb].
        Si es False, retorna los valores singulares como vector 1D (n,).

    Retorna:
    --------
    U : np.ndarray (n, n)
        Matriz ortogonal de vectores singulares izquierdos (matriz de características).
    Sigma : np.ndarray (n, n) o (n,)
        Valores singulares ordenados en orden descendente.
    Vt : np.ndarray (n, n)
        Matriz ortogonal V transpuesta (vectores singulares derechos transpuestos).
    """
    if matriz.ndim != 2:
        raise ValueError(f"Se esperaba una matriz bidimensional (n, n), pero se recibió dimensión {matriz.shape}")

    u, s, vt = np.linalg.svd(matriz, full_matrices=True)

    if retornar_sigma_diagonal:
        sigma = np.diag(s)
    else:
        sigma = s

    return u, sigma, vt


def generate_random_incremental_vector(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Genera el vector de incrementos aleatorios con valores continuos en el intervalo [0, 1),
    correspondiente a hr, hg o hb según las ecuaciones (8), (9) y (10) del artículo:
        h = (r_1, r_2, r_3, ..., r_n)

    Parámetros:
    -----------
    n : int
        Dimensión del vector (número de columnas de la matriz de características).
    seed : int, opcional
        Semilla para el generador de números aleatorios (útil para reproducibilidad experimental).

    Retorna:
    --------
    np.ndarray:
        Vector 1D de tamaño (n,) con números aleatorios uniformemente distribuidos en [0, 1).
    """
    if n <= 0:
        raise ValueError("La dimensión n debe ser un entero positivo mayor que cero.")

    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n)


def expand_matrix(matriz: np.ndarray, vector_random: np.ndarray) -> np.ndarray:
    """
    Expande una matriz de características ortogonales (U o V) al doble de filas (2n x n)
    intercalando entre cada fila original una fila incrementada con el vector aleatorio,
    según la metodología descrita en las ecuaciones (11), (12) y (13) del artículo:
        Fila 2i   (impar en base 1) : U[i, :]
        Fila 2i+1 (par en base 1)   : U[i, :] + vector_random

    Parámetros:
    -----------
    matriz : np.ndarray
        Matriz bidimensional de características (n, m), típicamente (n, n).
    vector_random : np.ndarray
        Vector 1D de incrementos aleatorios de tamaño (m,).

    Retorna:
    --------
    np.ndarray:
        Nueva matriz expandida de dimensiones (2n, m).
    """
    if matriz.ndim != 2:
        raise ValueError(f"La matriz a expandir debe ser bidimensional, se recibió: {matriz.shape}")

    n_filas, n_columnas = matriz.shape
    vector = np.asarray(vector_random).squeeze()

    if vector.ndim != 1 or vector.shape[0] != n_columnas:
        raise ValueError(
            f"El vector aleatorio debe tener longitud igual al número de columnas de la matriz ({n_columnas}), "
            f"pero tiene forma {vector.shape}"
        )

    # Creamos la matriz resultado de tamaño (2n, m)
    matriz_expandida = np.empty((2 * n_filas, n_columnas), dtype=matriz.dtype)

    # Filas pares (0, 2, 4, ... en Python / impares en el paper): características originales
    matriz_expandida[0::2, :] = matriz

    # Filas impares (1, 3, 5, ... en Python / pares en el paper): características incrementadas
    matriz_expandida[1::2, :] = matriz + vector

    return matriz_expandida


def expand_feature_pair(
    matriz_u: np.ndarray,
    matriz_vt: np.ndarray,
    vector_random: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de conveniencia que expande simultáneamente U y V:
    1. Expande U directamente a U_new (2n x n).
    2. Obtiene V = (Vt)^T, lo expande a V_new (2n x n) y retorna su transpuesta (V_new)^T (n x 2n).

    Parámetros:
    -----------
    matriz_u : np.ndarray
        Matriz ortogonal U (n, n).
    matriz_vt : np.ndarray
        Matriz ortogonal Vt = V^T (n, n).
    vector_random : np.ndarray
        Vector de incrementos aleatorios (n,).

    Retorna:
    --------
    u_new : np.ndarray (2n, n)
        Matriz U expandida.
    vt_new : np.ndarray (n, 2n)
        Matriz (V_new)^T expandida y transpuesta, lista para la reconstrucción.
    """
    u_new = expand_matrix(matriz_u, vector_random)

    # V es la transpuesta de Vt
    v = matriz_vt.T
    v_new = expand_matrix(v, vector_random)
    vt_new = v_new.T

    return u_new, vt_new


def reconstruct(
    matriz_sigma: np.ndarray,
    matriz_u: np.ndarray,
    matriz_v_transpuesta: np.ndarray,
    recortar_rango: bool = False
) -> np.ndarray:
    """
    Reconstruye un canal de color en alta resolución (2n x 2n) multiplicando las matrices
    expandidas según las ecuaciones (14), (15) y (16) del paper:
        B = U_new @ Sigma @ (V_new)^T

    Parámetros:
    -----------
    matriz_sigma : np.ndarray
        Matriz diagonal de valores singulares Sigma (n, n) o vector 1D de valores singulares (n,).
    matriz_u : np.ndarray
        Matriz de características izquierdas expandida U_new (2n, n).
    matriz_v_transpuesta : np.ndarray
        Matriz de características derechas expandida y transpuesta (n, 2n).
        Si se pasa sin transponer (2n, n), la función detecta las dimensiones y la transpone automáticamente.
    recortar_rango : bool, opcional (por defecto False)
        Si es True, recorta los valores resultantes al intervalo [0.0, 1.0].

    Retorna:
    --------
    np.ndarray:
        Canal de imagen reconstruido de dimensiones (2n, 2n).
    """
    # Si sigma se proporciona como vector 1D, convertir a matriz diagonal 2D
    if matriz_sigma.ndim == 1:
        sigma = np.diag(matriz_sigma)
    elif matriz_sigma.ndim == 2:
        sigma = matriz_sigma
    else:
        raise ValueError(f"matriz_sigma debe ser 1D o 2D, se recibió: {matriz_sigma.shape}")

    u = matriz_u
    vt = matriz_v_transpuesta

    # Si se pasó V_new (2n, n) en lugar de (V_new)^T (n, 2n), trasponerlo
    if vt.shape[0] == u.shape[0] and vt.shape[1] == sigma.shape[1]:
        vt = vt.T

    # Validar coherencia dimensional: (2n x n) @ (n x n) @ (n x 2n) = (2n x 2n)
    if u.shape[1] != sigma.shape[0] or sigma.shape[1] != vt.shape[0]:
        raise ValueError(
            f"Incompatibilidad de dimensiones en la multiplicación: "
            f"U: {u.shape}, Sigma: {sigma.shape}, V^T: {vt.shape}"
        )

    # Multiplicación matricial B = U @ Sigma @ V^T
    canal_reconstruido = u @ sigma @ vt

    if recortar_rango:
        canal_reconstruido = np.clip(canal_reconstruido, 0.0, 1.0)

    return canal_reconstruido