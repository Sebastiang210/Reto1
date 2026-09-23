import cv2
import numpy as np


class Preprocesador:
    """
    Se encarga de la extracción de FORMA y máscara de candidatos.
    En lugar de Canny sobre escala de grises (donde el papel de color
    se confunde con el fondo gris y el palito negro), usamos el espacio
    HSV (visto en las diapositivas 4 y 5):
    - Extraemos el canal S (Saturación), donde los colores vivos (rojo/verde)
      tienen valores altos (~180-250) y el fondo/palito negro/carro tienen
      valores bajos.
    - Aplicamos un umbral binario (cv2.threshold, visto en diapo 8).
    - Aplicamos dilatación morfológica (diapos 17-20) para cerrar contornos.
    - Encontramos contornos (diapos 22-25).
    """

    def __init__(self, tamano_kernel_morfologico=(5, 5)):
        self.kernel_morfologico = np.ones(tamano_kernel_morfologico, np.uint8)

    def obtener_mascara_color(self, frame, blur, umbral_saturacion):
        # Conversión a HSV (Diapositivas 4 y 5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Suavizado Gaussiano (Diapositivas 10-13)
        hsv_blur = cv2.GaussianBlur(hsv, (blur, blur), 0)

        # Canal Saturación (S): resalta los colores puros sobre fondos grises/negros
        canal_s = hsv_blur[:, :, 1]

        # Umbralización binaria (Diapositiva 8: cv2.threshold)
        _, binaria = cv2.threshold(canal_s, umbral_saturacion, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas: dilatación (Diapositivas 17-20)
        binaria_dilatada = cv2.dilate(binaria, self.kernel_morfologico, iterations=1)

        return canal_s, binaria_dilatada

    def encontrar_contornos(self, binaria):
        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

