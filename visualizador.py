import cv2
import numpy as np


class Visualizador:
    """
    Arma el mosaico 2x3 para ver, en una sola ventana, cada
    etapa del pipeline: original, grises, blur, bordes y el
    resultado final con las señales detectadas.
    """

    def __init__(self, ancho_celda=420, alto_celda=300):
        self.ancho_celda = ancho_celda
        self.alto_celda = alto_celda

    def crear_mosaico(self, frame, gris, gauss, bordes, salida):
        frame_r = self._redimensionar(frame)
        gris_r = self._redimensionar(self._gris_a_bgr(gris))
        gauss_r = self._redimensionar(self._gris_a_bgr(gauss))
        bordes_r = self._redimensionar(self._gris_a_bgr(bordes))
        salida_r = self._redimensionar(salida)
        negro = np.zeros_like(frame_r)

        fila1 = np.hstack((frame_r, gris_r, gauss_r))
        fila2 = np.hstack((bordes_r, salida_r, negro))
        panel = np.vstack((fila1, fila2))

        return self._agregar_titulos(panel)

    def _redimensionar(self, imagen):
        return cv2.resize(imagen, (self.ancho_celda, self.alto_celda))

    def _gris_a_bgr(self, imagen):
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    def _agregar_titulos(self, panel):
        titulos = [
            ("Original", (20, 30)),
            ("Grises", (440, 30)),
            ("Gaussian Blur", (860, 30)),
            ("Canny + Dilatacion", (20, 330)),
            ("PARE / SIGA detectados", (440, 330)),
        ]

        for texto, posicion in titulos:
            cv2.putText(
                panel, texto, posicion,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        cv2.putText(
            panel, "Q para salir", (860, 330),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        return panel
