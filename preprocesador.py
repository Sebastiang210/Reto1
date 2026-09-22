import cv2
import numpy as np


class Preprocesador:
    """
    Se encarga únicamente de la parte de FORMA del pipeline:
    convierte el frame de la cámara en una lista de contornos,
    usando exactamente lo visto en las diapositivas:

        gris -> blur gaussiano -> Canny -> dilatación -> contornos

    Esta clase no sabe nada de colores ni de PARE/SIGA; solo
    entrega figuras (contornos) para que otra clase decida qué
    son.
    """

    def __init__(self, tamano_kernel_morfologico=(3, 3)):
        # Kernel para la dilatación (operación morfológica).
        self.kernel_morfologico = np.ones(tamano_kernel_morfologico, np.uint8)

    def a_gris_y_blur(self, frame, blur):
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gris, (blur, blur), 0)
        return gris, gauss

    def detectar_bordes(self, gauss, canny_bajo, canny_alto):
        bordes = cv2.Canny(gauss, canny_bajo, canny_alto)

        # Dilatación: engrosa los bordes para cerrar pequeños
        # huecos, así el contorno queda cerrado y findContours
        # lo puede tomar como una sola figura.
        bordes = cv2.dilate(bordes, self.kernel_morfologico, iterations=1)

        return bordes

    def encontrar_contornos(self, bordes):
        contornos, _ = cv2.findContours(
            bordes,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos
