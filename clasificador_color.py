import cv2
import numpy as np


class ClasificadorColor:
    """
    Decide si un contorno es PARE (rojo) o ADELANTE (verde)
    usando K-Means sobre los píxeles interiores del contorno
    en el espacio HSV (diapositivas 4 y 5 del PDF).
    K-Means extrae los centroides de color dominante,
    separando el color del fondo del texto blanco o reflejos.
    """

    def __init__(self, minimo_pixeles=30):
        self.minimo_pixeles = minimo_pixeles
        self.criterios_kmeans = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

    def _mascara_del_contorno(self, forma_frame, contorno):
        mascara = np.zeros(forma_frame, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        return mascara

    def _pixeles_hsv_dentro(self, frame_bgr, mascara):
        # AND entre frame y máscara (Diapositiva 7)
        recorte_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mascara)
        # Transformación BGR a HSV (Diapositivas 4 y 5)
        recorte_hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
        return recorte_hsv[mascara == 255].astype(np.float32)

    def _color_dominante_hsv(self, pixeles_hsv):
        # K-Means con k=2 para separar el fondo de letras o reflejos
        k = 2 if len(pixeles_hsv) >= 30 else 1
        _, etiquetas, centros = cv2.kmeans(
            pixeles_hsv,
            k,
            None,
            self.criterios_kmeans,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        conteos = np.bincount(etiquetas.flatten())
        # Tomamos el cluster con más píxeles (color dominante de la señal)
        indice_dominante = np.argmax(conteos)
        return centros[indice_dominante]

    def clasificar(self, frame_bgr, contorno_aproximado):
        """
        Devuelve ("PARE" | "ADELANTE" | "DESCONOCIDO", (h, s, v))
        o (None, None) si no hay suficientes píxeles.
        """
        mascara = self._mascara_del_contorno(frame_bgr.shape[:2], contorno_aproximado)
        pixeles_hsv = self._pixeles_hsv_dentro(frame_bgr, mascara)

        if len(pixeles_hsv) < self.minimo_pixeles:
            return None, None

        h, s, v = self._color_dominante_hsv(pixeles_hsv)

        # Si la saturación del cluster dominante es muy baja, es blanco, gris o negro
        if s < 40 or v < 30:
            return "DESCONOCIDO", (h, s, v)

        # En OpenCV HSV (H de 0 a 179):
        # Rojo: extremos 0-15 y 165-179
        # Verde: 35 a 90
        if h <= 15 or h >= 165:
            return "PARE", (h, s, v)
        elif 35 <= h <= 90:
            return "ADELANTE", (h, s, v)

        return "DESCONOCIDO", (h, s, v)
