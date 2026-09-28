import cv2


class DetectorSenales:
    """
    Combina el Preprocesador (máscara de color) con el
    ClasificadorColor (K-Means) para encontrar las señales
    PARE/SIGA entre los contornos de la máscara, y las dibuja
    sobre el frame.

    Un contorno se acepta como señal si:
      - su área supera el mínimo (el cartel está cerca),
      - no toca el borde del frame (un cartel visto a medias no
        deja confirmar su forma),
      - su aproximación poligonal tiene entre `lados_min` y
        `lados_max` vértices (un octágono da 8; se deja un
        margen porque approxPolyDP a veces da 7 o 9),
      - su relación de aspecto es cercana a 1 (el octágono es
        tan ancho como alto; un cartel cortado por el borde
        del frame no lo es),
      - es sólido (área / área de su envolvente convexa), o sea
        una figura convexa y no una mancha irregular.
    Después K-Means decide si el interior es rojo o verde.
    """

    COLORES_DIBUJO = {
        "PARE": (0, 0, 255),        # rojo en BGR
        "SIGA": (0, 200, 0),        # verde en BGR
        "DESCONOCIDO": (0, 255, 255)  # amarillo: forma válida pero color dudoso
    }

    ASPECTO_MIN = 0.6
    ASPECTO_MAX = 1.6
    SOLIDEZ_MIN = 0.85
    MARGEN_BORDE = 3

    def __init__(self, clasificador_color):
        self.clasificador_color = clasificador_color

    def procesar(self, frame, contornos, parametros):
        """
        Recibe el frame original, los contornos de la máscara de
        color y los parámetros de los trackbars. Devuelve
        (frame_dibujado, detecciones), donde detecciones es una
        lista de diccionarios {"etiqueta", "area", "contorno"}.
        """
        salida = frame.copy()
        detecciones = []

        for contorno in contornos:
            aproximacion = self._aproximar_si_vale(contorno, frame.shape, parametros)

            if aproximacion is None:
                continue

            etiqueta, _hsv = self.clasificador_color.clasificar(frame, aproximacion)

            if etiqueta is None:
                continue

            detecciones.append({
                "etiqueta": etiqueta,
                "area": cv2.contourArea(contorno),
                "contorno": aproximacion,
            })
            self._dibujar(salida, aproximacion, etiqueta)

        return salida, detecciones

    def _aproximar_si_vale(self, contorno, forma_frame, parametros):
        area = cv2.contourArea(contorno)
        if area < parametros["area_minima"]:
            return None

        if self._toca_borde(contorno, forma_frame):
            return None

        perimetro = cv2.arcLength(contorno, True)
        epsilon = (parametros["precision"] / 100) * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)

        # Forma: PARE y SIGA son octágonos. Cualquier otra
        # figura se descarta acá.
        if not parametros["lados_min"] <= len(aproximacion) <= parametros["lados_max"]:
            return None

        _, _, ancho, alto = cv2.boundingRect(aproximacion)
        aspecto = ancho / alto
        if not self.ASPECTO_MIN <= aspecto <= self.ASPECTO_MAX:
            return None

        area_envolvente = cv2.contourArea(cv2.convexHull(contorno))
        if area_envolvente == 0 or area / area_envolvente < self.SOLIDEZ_MIN:
            return None

        return aproximacion

    def _toca_borde(self, contorno, forma_frame):
        alto, ancho = forma_frame[:2]
        x, y, w, h = cv2.boundingRect(contorno)
        m = self.MARGEN_BORDE
        return x <= m or y <= m or x + w >= ancho - m or y + h >= alto - m

    def _dibujar(self, salida, aproximacion, etiqueta):
        x, y, w, h = cv2.boundingRect(aproximacion)
        color = self.COLORES_DIBUJO[etiqueta]

        cv2.drawContours(salida, [aproximacion], -1, color, 3)
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)

        y_texto = y - 10 if y - 10 > 10 else y + h + 25
        cv2.putText(
            salida,
            f"{etiqueta} ({len(aproximacion)} lados)",
            (x, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
