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
        # Usamos k=3 clusters para separar:
        # 1) Color principal (rojo o verde de la señal)
        # 2) Letras/brillos/blanco
        # 3) Sombras/negro o bordes
        k = 3 if len(pixeles) >= 30 else 2
        _, etiquetas, centros = cv2.kmeans(
            pixeles,
            k,
            None,
            self.criterios_kmeans,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        conteos = np.bincount(etiquetas.flatten())
        # Ordenamos los clusters de mayor a menor cantidad de píxeles
        indices_ordenados = np.argsort(-conteos)

        # Buscamos entre los clusters aquel que tenga color significativo (no sea negro ni blanco puro)
        for idx in indices_ordenados:
            b, g, r = centros[idx]
            etiqueta = self._nombre_por_rgb(r, g, b)
            if etiqueta in ("PARE", "ADELANTE"):
                return centros[idx], etiqueta

        # Si ninguno califica como PARE o ADELANTE, devolvemos el más abundante
        indice_dominante = indices_ordenados[0]
        b, g, r = centros[indice_dominante]
        return centros[indice_dominante], self._nombre_por_rgb(r, g, b)

    def clasificar(self, frame_bgr, contorno_aproximado):
        """
        Devuelve ("PARE" | "ADELANTE" | "DESCONOCIDO", (r, g, b))
        o (None, None) si no hay píxeles suficientes.
        """
        mascara = self._mascara_del_contorno(frame_bgr.shape[:2], contorno_aproximado)
        pixeles = self._pixeles_bgr_dentro(frame_bgr, mascara)

        if len(pixeles) < self.minimo_pixeles:
            return None, None

        centro_elegido, etiqueta = self._color_dominante(pixeles)
        b, g, r = centro_elegido

        return etiqueta, (r, g, b)

    def _nombre_por_rgb(self, r, g, b):
        """
        Clasificación en espacio RGB:
        - Descarta negro/sombra (muy baja intensidad)
        - Descarta blanco/gris (r, g, b muy similares)
        - Descarta el azul del chasis del carro (b > r y b > g)
        - Detecta rojo (PARE) y verde (ADELANTE)
        """
        brillo = (r + g + b) / 3.0

        # 1. Descartar sombras muy oscuras o negros
        if brillo < 35:
            return "DESCONOCIDO"

        # 2. Descartar blancos o reflejos intensos donde R, G, B están saturados
        if r > 200 and g > 200 and b > 200:
            return "DESCONOCIDO"

        # 3. Descartar el chasis azul del robot (donde el canal Azul predomina)
        if b > r + 15 and b > g + 15:
            return "DESCONOCIDO"

        # 4. Escala de ROJO (PARE):
        # R debe ser superior a G y B por margen claro
        if r > 70 and r > g * 1.25 and r > b * 1.2:
            return "PARE"
        if (r - g > 25) and (r - b > 25):
            return "PARE"

        # 5. Escala de VERDE (ADELANTE):
        # G debe ser superior a R y B por margen claro
        if g > 55 and g > r * 1.15 and g > b * 1.1:
            return "ADELANTE"
        if (g - r > 20) and (g - b > 15):
            return "ADELANTE"

        return "DESCONOCIDO"
