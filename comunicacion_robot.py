import config_robot


class Accion:
    """
    Nombres de las acciones que el programa le puede pedir al
    robot. Son solo nombres: la letra real de cada una vive en
    config_robot.LETRAS.
    """
    ADELANTE = "ADELANTE"
    IZQUIERDA = "IZQUIERDA"
    DERECHA = "DERECHA"
    PARAR = "PARAR"


class SalidaConsola:
    """Canal de prueba: imprime la letra en vez de enviarla."""

    def escribir(self, texto, accion):
        print(f"[robot] {accion:<9} -> {texto!r}")

    def cerrar(self):
        pass


class SalidaSerial:
    """
    Canal real: envía la letra por el puerto serial (cable USB
    o módulo Bluetooth emparejado como puerto COM). Necesita
    la librería pyserial (pip install pyserial).
    """

    def __init__(self, puerto, baudios):
        import serial
        self.conexion = serial.Serial(puerto, baudios, timeout=1)

    def escribir(self, texto, accion):
        self.conexion.write(texto.encode("ascii"))

    def cerrar(self):
        self.conexion.close()


class ComunicacionRobot:
    """
    Traduce una Accion a su letra (según config_robot) y la
    envía por el canal configurado. Solo envía cuando la
    acción cambia, para no saturar al robot mandando la misma
    letra en cada frame.
    """

    def __init__(self, letras=None, salida=None):
        self.letras = letras or config_robot.LETRAS
        self.salida = salida or self._crear_salida()
        self.ultima_accion = None

    def _crear_salida(self):
        if config_robot.PUERTO_SERIAL:
            return SalidaSerial(config_robot.PUERTO_SERIAL, config_robot.BAUDIOS)
        return SalidaConsola()

    def enviar(self, accion):
        if accion == self.ultima_accion:
            return

        texto = self.letras[accion] + config_robot.TERMINADOR
        self.salida.escribir(texto, accion)
        self.ultima_accion = accion

    def cerrar(self):
        # Antes de soltar la conexión se deja el robot quieto.
        self.enviar(Accion.PARAR)
        self.salida.cerrar()
