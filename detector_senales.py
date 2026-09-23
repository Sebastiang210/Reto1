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
        h_frame, w_frame = frame.shape[:2]
        max_area = (h_frame * w_frame) * 0.65  # Ignorar si cubre más del 65% de la pantalla

        for contorno in contornos:
            aproximacion = self._aproximar_si_vale(contorno, area_minima, precision, max_area)

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

    # Lados válidos para prueba (cuadrados/rectángulos=4, octágonos=8)
    LADOS_VALIDOS = (4, 8)

    def _aproximar_si_vale(self, contorno, area_minima, precision, max_area=150000):
        area = cv2.contourArea(contorno)
        if area < area_minima or area > max_area:
            return None

        # 1. Filtro de solidez (Área del contorno / Área de su envolvente convexa)
        # Una figura sólida geométrica da solidez > 0.82; pilas o manos dan valores mucho menores.
        envolvente = cv2.convexHull(contorno)
        area_envolvente = cv2.contourArea(envolvente)
        if area_envolvente == 0:
            return None
        solidez = float(area) / area_envolvente
        if solidez < 0.80:
            return None

        # 2. Filtro de relación de aspecto (bounding box)
        # Descarta cables o franjas alargadas (diapositiva 25: cv2.boundingRect)
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return None

        # 3. Aproximación poligonal (diapositivas 23-24: cv2.approxPolyDP)
        perimetro = cv2.arcLength(contorno, True)
        epsilon = (precision / 100) * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)

        # Se aceptan polígonos de 4 u 8 lados
        if len(aproximacion) not in self.LADOS_VALIDOS:
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
