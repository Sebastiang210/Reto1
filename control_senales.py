from comunicacion_robot import Accion


class ControlSenales:
    """
    Máquina de estados que decide si el robot obedece al
    seguidor de línea o se queda quieto por una señal.

        SIGUIENDO --(PARE confirmado)--> DETENIDO
        DETENIDO  --(SIGA confirmado)--> SIGUIENDO

    - En SIGUIENDO se envía la acción del seguidor de línea.
    - En DETENIDO se envía PARAR, sin importar la línea.
    - Una señal cuenta como "confirmada" cuando aparece varios
      frames seguidos (evita que un falso positivo de un solo
      frame frene al robot).
    - Al reanudar con SIGA, se ignora PARE durante un tiempo de
      espera, para que el mismo cartel (si sigue a la vista) no
      lo vuelva a frenar enseguida.
    """

    SIGUIENDO = "SIGUIENDO"
    DETENIDO = "DETENIDO"

    def __init__(self):
        self.estado = self.SIGUIENDO
        self.senal_anterior = None
        self.frames_senal = 0
        self.frames_espera = 0

    def actualizar(self, detecciones, accion_linea, parametros):
        """
        Recibe las detecciones del frame (ver
        DetectorSenales.procesar), la acción que propone el
        seguidor de línea y los parámetros de los trackbars.
        Devuelve la acción que hay que enviarle al robot.
        """
        senal = self._senal_confirmada(detecciones, parametros["frames_confirmacion"])

        if self.frames_espera > 0:
            self.frames_espera -= 1

        if self.estado == self.SIGUIENDO:
            if senal == "PARE" and self.frames_espera == 0:
                self.estado = self.DETENIDO
        elif senal == "SIGA":
            self.estado = self.SIGUIENDO
            self.frames_espera = parametros["frames_espera"]

        if self.estado == self.DETENIDO:
            return Accion.PARAR
        return accion_linea

    def _senal_confirmada(self, detecciones, frames_confirmacion):
        # Si en el frame hay varias señales, manda la más grande
        # (la más cercana al robot). Las de color dudoso no
        # cuentan.
        validas = [d for d in detecciones if d["etiqueta"] in ("PARE", "SIGA")]
        senal = max(validas, key=lambda d: d["area"])["etiqueta"] if validas else None

        if senal is not None and senal == self.senal_anterior:
            self.frames_senal += 1
        else:
            self.frames_senal = 1 if senal is not None else 0
        self.senal_anterior = senal

        if senal is not None and self.frames_senal >= frames_confirmacion:
            return senal
        return None
