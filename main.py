import cv2

from interfaz_control import InterfazControl
from preprocesador import Preprocesador
from clasificador_color import ClasificadorColor
from detector_senales import DetectorSenales
from visualizador import Visualizador


# ============================================================
# IDEA GENERAL (misma que antes, ahora repartida en clases)
# ============================================================
# Las señales PARE y SIGA tienen la MISMA forma (octágono, 8
# lados), así que la forma sola no alcanza para diferenciarlas.
# El programa junta dos piezas:
#
#   Preprocesador     -> FORMA: gris, blur, Canny, dilatación,
#                         contornos (todo visto en clase).
#   ClasificadorColor -> COLOR: K-Means sobre los píxeles de
#                         adentro de cada contorno, para decidir
#                         si es rojo (PARE) o verde (SIGA).
#   DetectorSenales    -> junta las dos piezas de arriba: filtra
#                         los contornos que son octágonos y les
#                         pregunta su color.
#   Visualizador        -> arma el mosaico para ver cada etapa.
#   InterfazControl      -> ventana + trackbars.
#
# Esta clase (AplicacionDetector) es la que arma todas las
# piezas y corre el loop de la cámara.


class AplicacionDetector:
    """
    Punto de entrada del programa: abre la cámara, arma cada
    pieza del pipeline y corre el loop principal (leer frame,
    procesarlo, mostrarlo) hasta que el usuario presione 'q'.
    """

    def __init__(self, indice_camara=0):
        self.cap = cv2.VideoCapture(indice_camara)

        self.interfaz = InterfazControl()
        self.preprocesador = Preprocesador()
        self.clasificador_color = ClasificadorColor()
        self.detector = DetectorSenales(self.clasificador_color)
        self.visualizador = Visualizador()

    def ejecutar(self):
        if not self.cap.isOpened():
            print("No se pudo abrir la camara.")
            return

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("No se pudo leer el frame.")
                    break

                panel = self._procesar_frame(frame)
                cv2.imshow(self.interfaz.nombre_ventana, panel)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def _procesar_frame(self, frame):
        canny_bajo, canny_alto, blur, area_minima, precision = (
            self.interfaz.leer_controles()
        )

        gris, gauss = self.preprocesador.a_gris_y_blur(frame, blur)
        bordes = self.preprocesador.detectar_bordes(gauss, canny_bajo, canny_alto)
        contornos = self.preprocesador.encontrar_contornos(bordes)

        salida = self.detector.procesar(frame, contornos, area_minima, precision)

        return self.visualizador.crear_mosaico(frame, gris, gauss, bordes, salida)


if __name__ == "__main__":
    AplicacionDetector().ejecutar()
