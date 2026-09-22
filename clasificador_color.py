import cv2
import numpy as np


class ClasificadorColor:
    """
    Decide de qué color es el interior de un contorno, usando
    K-Means sobre los píxeles que quedan dentro de él.

    K-Means no es parte de las diapositivas: se usa acá nada
    más para esta decisión puntual (separar el color de fondo
    del octágono del blanco de las letras "PARE"/"SIGA"). La
    FORMA se sigue detectando solo con las herramientas de
    clase (eso vive en Preprocesador).
    """

    def __init__(self, minimo_pixeles=30):
        self.minimo_pixeles = minimo_pixeles

        # Criterio de parada de K-Means: parar a las 10
        # iteraciones o antes si el cambio entre iteraciones es
        # menor a 1.0.
        self.criterios_kmeans = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

    def clasificar(self, frame_bgr, contorno_aproximado):
        """
        Devuelve ("PARE" | "ADELANTE" | "DESCONOCIDO", (r, g, b))
        o (None, None) si no hay píxeles suficientes.
        """
        mascara = self._mascara_del_contorno(frame_bgr.shape[:2], contorno_aproximado)
        pixeles = self._pixeles_bgr_dentro(frame_bgr, mascara)

        if len(pixeles) < self.minimo_pixeles:
            return None, None

        b, g, r = self._color_dominante(pixeles)
        etiqueta = self._nombre_por_rgb(r, g, b)

        return etiqueta, (r, g, b)

    def _mascara_del_contorno(self, forma_frame, contorno):
        # Imagen negra del tamaño del frame, rellena de blanco
        # (255) solo en la región del contorno.
        mascara = np.zeros(forma_frame, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        return mascara

    def _pixeles_bgr_dentro(self, frame_bgr, mascara):
        # Operación AND: el frame AND la máscara deja pasar
        # únicamente los píxeles de adentro del contorno.
        recorte = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mascara)
        return recorte[mascara == 255].astype(np.float32)

    def _color_dominante(self, pixeles):
        # k=2: separa el color principal del contorno de texto/brillos
        k = 2
        _, etiquetas, centros = cv2.kmeans(
            pixeles,
            k,
            None,
            self.criterios_kmeans,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        conteos = np.bincount(etiquetas.flatten())
        indice_dominante = np.argmax(conteos)

        return centros[indice_dominante]

    def _nombre_por_rgb(self, r, g, b):
        """
        Clasificación directa en espacio RGB/BGR:
        - Escala de Rojo: el componente R predomina significativamente sobre G y B.
        - Escala de Verde: el componente G predomina significativamente sobre R y B.
        """
        # Descartar colores oscuros o muy grises/blancos donde R, G y B son casi idénticos
        diferencia_rg = r - g
        diferencia_gr = g - r

        # Predominio de rojo: R mayor que G y B
        if r > 60 and r > g * 1.2 and r > b * 1.2:
            return "PARE"

        # Predominio de verde: G mayor que R y B
        if g > 50 and g > r * 1.15 and g > b * 1.1:
            return "ADELANTE"

        # Criterio de respaldo si la iluminación desbalancea los brillos
        if diferencia_rg > 25 and r > b:
            return "PARE"
        elif diferencia_gr > 20 and g > b:
            return "ADELANTE"

        return "DESCONOCIDO"
