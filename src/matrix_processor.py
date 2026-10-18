"""
Módulo de procesamiento matricial para el esquema SVD + Interpolación.
Implementa la factorización SVD, generación de vectores de incremento aleatorio,
expansión de matrices de características ortogonales mediante Splines Cúbicos
asistidos por números aleatorios y reconstrucción del canal en alta resolución.
"""

from typing import Tuple, Union, Optional
import numpy as np
from scipy.interpolate import CubicSpline


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
        Dimensión del vector (número de filas/columnas de la matriz de características).
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


def expand_matrix(
    matriz: np.ndarray,
    vector_random: np.ndarray,
    metodo: str = "spline"
) -> np.ndarray:
    """
    Expande una matriz de características ortogonales (U o V) al doble de filas (2n x n).
    
    Por defecto ('spline'), interpola cada columna de la matriz utilizando Splines Cúbicos
    evaluados en las posiciones intermedias definidas por el vector de incrementos aleatorios,
    cumpliendo con la premisa del paper: "Only one increment point has been used in the middle
    of any two continuous points to enlarge the image into double size".

    Las filas pares (0, 2, 4, ... en Python / impares en el paper) preservan exactamente
    las características de la matriz original, y las filas impares (1, 3, 5, ...) contienen
    los valores interpolados mediante Spline Cúbico.

    Parámetros:
    -----------
    matriz : np.ndarray
        Matriz bidimensional de características (n, m), típicamente (n, n).
    vector_random : np.ndarray
        Vector 1D de incrementos aleatorios en [0, 1).
    metodo : str, opcional ('spline' o 'literal')
        - 'spline': Interpolación por Splines Cúbicos con paso aleatorio (Recomendado).
        - 'literal': Adición directa U + h de la Ec. 11 (genera artefactos por overflow de energía).

    Retorna:
    --------
    np.ndarray:
        Nueva matriz expandida de dimensiones (2n, m).
    """
    if matriz.ndim != 2:
        raise ValueError(f"La matriz a expandir debe ser bidimensional, se recibió: {matriz.shape}")

    n_filas, n_columnas = matriz.shape
    vector = np.asarray(vector_random).squeeze()

    if vector.ndim != 1 or vector.shape[0] != n_filas:
        # En matrices cuadradas n_filas == n_columnas, pero verificamos dimensión n
        if vector.shape[0] != n_columnas:
            raise ValueError(
                f"El vector aleatorio debe tener tamaño compatible con la matriz ({n_filas}), "
                f"pero tiene forma {vector.shape}"
            )

    if metodo.lower() == "literal":
        # Implementación literal Ec. 11 (produce artefactos numéricos por escala de U)
        matriz_expandida = np.empty((2 * n_filas, n_columnas), dtype=matriz.dtype)
        matriz_expandida[0::2, :] = matriz
        matriz_expandida[1::2, :] = matriz + vector
        return matriz_expandida

    # Método 'spline': Interpolación por Splines Cúbicos asistida por números aleatorios
    x_nodos = np.arange(n_filas, dtype=np.float64)

    # El paso incremental entre x_i y x_{i+1} se ubica en el intervalo (0.1, 0.9)
    # según el vector aleatorio, situándolo en la región intermedia entre ambos nodos
    r_paso = 0.2 + 0.6 * np.clip(vector, 0.0, 1.0)

    x_eval = np.empty(2 * n_filas, dtype=np.float64)
    # Filas pares: nodos originales exactos (conserva características originales)
    x_eval[0::2] = x_nodos
    # Filas impares: puntos intermedios interpolados
    x_eval[1:-1:2] = x_nodos[:-1] + r_paso[:-1]
    # Último punto: extrapolación suave dentro del último tramo
    x_eval[-1] = x_nodos[-1] + 0.5 * r_paso[-1]

    # Ajustar Spline Cúbico por columnas (axis=0)
    cs = CubicSpline(x_nodos, matriz, axis=0, bc_type="natural")
    matriz_expandida = cs(x_eval)

    return matriz_expandida


def expand_feature_pair(
    matriz_u: np.ndarray,
    matriz_vt: np.ndarray,
    vector_random: np.ndarray,
    metodo: str = "spline"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función que expande simultáneamente U y V:
    1. Expande U directamente a U_new (2n x n).
    2. Obtiene V = (Vt)^T, lo expande a V_new (2n x n) y retorna su transpuesta (V_new)^T (n x 2n).

    Parámetros:
    -----------
    matriz_u : np.ndarray
        Matriz ortogonal U (n, n).
    matriz_vt : np.ndarray
        Matriz ortogonal Vt = V^T (n, n).
    vector_random : np.ndarray
        Vector de incrementos aleatorios.
    metodo : str, opcional ('spline' o 'literal')

    Retorna:
    --------
    u_new : np.ndarray (2n, n)
        Matriz U expandida.
    vt_new : np.ndarray (n, 2n)
        Matriz (V_new)^T expandida y transpuesta, lista para la reconstrucción.
    """
    u_new = expand_matrix(matriz_u, vector_random, metodo=metodo)

    # V es la transpuesta de Vt
    v = matriz_vt.T
    v_new = expand_matrix(v, vector_random, metodo=metodo)
    vt_new = v_new.T

    return u_new, vt_new


def reconstruct(
    matriz_sigma: np.ndarray,
    matriz_u: np.ndarray,
    matriz_v_transpuesta: np.ndarray,
    recortar_rango: bool = True
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
    recortar_rango : bool, opcional (por defecto True)
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