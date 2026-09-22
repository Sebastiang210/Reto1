import cv2


class DetectorSenales:
    """
    Combina el Preprocesador (forma) con el ClasificadorColor
    (color) para encontrar las señales PARE/SIGA entre una
    lista de contornos, y las dibuja sobre el frame.
    """

    COLORES_DIBUJO = {
        "PARE": (0, 0, 255),        # rojo en BGR
        "ADELANTE": (0, 200, 0),    # verde en BGR
        "DESCONOCIDO": (0, 255, 255)  # amarillo: es octágono pero color dudoso
    }

    LADOS_OCTAGONO = 8

    def __init__(self, clasificador_color):
        self.clasificador_color = clasificador_color

    def procesar(self, frame, contornos, area_minima, precision):
        """
        Recibe el frame original y los contornos ya encontrados
        por el Preprocesador, y devuelve una copia del frame con
        las señales detectadas dibujadas encima.
        """
        salida = frame.copy()

        for contorno in contornos:
            aproximacion = self._aproximar_si_vale(contorno, area_minima, precision)

            if aproximacion is None:
                continue

            etiqueta, _hsv = self.clasificador_color.clasificar(frame, aproximacion)

            if etiqueta is None:
                continue

            # Imprimir en consola según lo solicitado
            if etiqueta == "PARE":
                print("pare")
            elif etiqueta == "ADELANTE":
                print("adelante")

            self._dibujar(salida, aproximacion, etiqueta)

        return salida

    def _aproximar_si_vale(self, contorno, area_minima, precision):
        area = cv2.contourArea(contorno)
        if area < area_minima:
            return None

        perimetro = cv2.arcLength(contorno, True)
        epsilon = (precision / 100) * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)

        # Solo nos interesan los octágonos: PARE y SIGA son
        # octágonos. Cualquier otra figura se descarta acá.
        if len(aproximacion) != self.LADOS_OCTAGONO:
            return None

        return aproximacion

    def _dibujar(self, salida, aproximacion, etiqueta):
        x, y, w, h = cv2.boundingRect(aproximacion)
        color = self.COLORES_DIBUJO[etiqueta]

        cv2.drawContours(salida, [aproximacion], -1, color, 3)
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)

        y_texto = y - 10 if y - 10 > 10 else y + h + 25
        cv2.putText(
            salida,
            etiqueta,
            (x, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA
        )
