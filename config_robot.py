# ============================================================
# CONFIGURACIÓN DEL ROBOT
# ============================================================
# Este es el ÚNICO archivo que hay que tocar cuando en clase
# digan qué letra entiende el robot para cada acción, o por
# qué puerto se conecta. El resto del programa trabaja con los
# nombres de las acciones ("ADELANTE", "IZQUIERDA", ...) y
# nunca con las letras directamente.

# Letra que se envía al robot por cada acción.
LETRAS = {
    "ADELANTE": "F",
    "IZQUIERDA": "I",
    "DERECHA": "D",
    "PARAR": "S",
}

# Puerto serial del robot (por ejemplo "COM3" en Windows o
# "/dev/ttyUSB0" en Linux). Con None las letras solo se
# imprimen en consola, útil para probar con los videos.
PUERTO_SERIAL = None
BAUDIOS = 9600

# Caracter que se agrega después de cada letra (por ejemplo
# "\n" si el robot lee línea por línea). Vacío = solo la letra.
TERMINADOR = ""
