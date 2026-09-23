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
    Punto de entrada del programa: abre el archivo de video o cámara,
    arma cada pieza del pipeline y corre el loop principal (leer frame,
    procesarlo, mostrarlo) hasta que el usuario presione 'q'.
    """

    def __init__(self, fuente_video="uploads/ideal/video1.mp4"):
        self.fuente_video = fuente_video
        self.cap = cv2.VideoCapture(fuente_video)

        self.interfaz = InterfazControl()
        self.preprocesador = Preprocesador()
        self.clasificador_color = ClasificadorColor()
        self.detector = DetectorSenales(self.clasificador_color)
        self.visualizador = Visualizador()

    def ejecutar(self):
        if not self.cap.isOpened():
            print(f"No se pudo abrir el video/camara: {self.fuente_video}")
            return

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    # Si finaliza el video, reiniciar al inicio para bucle
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Fin del video.")
                        break

                panel = self._procesar_frame(frame)
                cv2.imshow(self.interfaz.nombre_ventana, panel)

                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def _procesar_frame(self, frame):
        umbral_sat, blur, area_minima, precision = (
            self.interfaz.leer_controles()
        )

        canal_s, binaria = self.preprocesador.obtener_mascara_color(frame, blur, umbral_sat)
        contornos = self.preprocesador.encontrar_contornos(binaria)

        salida = self.detector.procesar(frame, contornos, area_minima, precision)

        return self.visualizador.crear_mosaico(frame, canal_s, canal_s, binaria, salida)


if __name__ == "__main__":
    import sys
    ruta_video = sys.argv[1] if len(sys.argv) > 1 else "uploads/ideal/video1.mp4"
    AplicacionDetector(fuente_video=ruta_video).ejecutar()

