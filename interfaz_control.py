import cv2


class InterfazControl:
    """
    Encapsula la ventana de OpenCV y los trackbars (controles
    deslizantes) que permiten ajustar los parámetros del
    pipeline en vivo, sin tener que tocar el código cada vez.
    """

    def __init__(self, nombre_ventana="Detector PARE - SIGA"):
        self.nombre_ventana = nombre_ventana
        cv2.namedWindow(self.nombre_ventana, cv2.WINDOW_NORMAL)
        self._crear_trackbars()

    def _nada(self, valor):
        # Callback vacío que exige la API de trackbars de OpenCV
        # (siempre espera una función, aunque no haga nada).
        pass

    def _crear_trackbars(self):
        cv2.createTrackbar("Umbral Saturacion", self.nombre_ventana, 90, 255, self._nada)
        cv2.createTrackbar("Gaussian Blur", self.nombre_ventana, 5, 31, self._nada)
        cv2.createTrackbar("Area Minima", self.nombre_ventana, 2000, 40000, self._nada)
        cv2.createTrackbar("Precision Poligono", self.nombre_ventana, 3, 20, self._nada)

    def leer_controles(self):
        """
        Devuelve los valores actuales de los trackbars, ya
        validados (kernel de blur impar, precisión mínima 1).
        """
        umbral_sat = cv2.getTrackbarPos("Umbral Saturacion", self.nombre_ventana)
        blur = cv2.getTrackbarPos("Gaussian Blur", self.nombre_ventana)
        area_minima = cv2.getTrackbarPos("Area Minima", self.nombre_ventana)
        precision = cv2.getTrackbarPos("Precision Poligono", self.nombre_ventana)

        # El kernel del Gaussian Blur tiene que ser impar y >= 1
        if blur < 1:
            blur = 1
        if blur % 2 == 0:
            blur += 1

        if precision < 1:
            precision = 1

        return umbral_sat, blur, area_minima, precision
