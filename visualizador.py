import cv2
import numpy as np


class Visualizador:
    """
    Muestra la salida final con las señales detectadas.
    """

    def __init__(self, ancho_celda=640, alto_celda=480):
        self.ancho_celda = ancho_celda
        self.alto_celda = alto_celda

    def crear_mosaico(self, frame, gris, gauss, bordes, salida):
        return self._redimensionar(salida)

    def _redimensionar(self, imagen):
        return cv2.resize(imagen, (self.ancho_celda, self.alto_celda))

