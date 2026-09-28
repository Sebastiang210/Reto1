# Reto 1: Seguidor de línea con señales PARE / SIGA

Programa de visión artificial para un robot móvil que sigue una línea negra sobre la pista, corrige su trayectoria y obedece dos señales: un octágono rojo (PARE) y uno verde (SIGA). No usa redes neuronales ni modelos entrenados. Todo está hecho con operaciones vistas en el curso: recortes de ROI, espacios de color, umbralización, morfología, contornos, aproximación poligonal y K-Means básico.

## Cómo correrlo

```bash
pip install opencv-python numpy
python main.py                              # video por defecto (uploads/ideal/video1.mp4)
python main.py uploads/ideal/video3.mp4     # otro video
python main.py 0                            # cámara 0
```

Se abren tres ventanas: el video procesado (con la máscara de color al lado) y dos ventanas de trackbars para calibrar en vivo. El significado de cada trackbar se imprime en consola al arrancar. Con `q` se cierra.

## Comunicación con el robot

El robot recibe una letra por cada acción. Las letras y el puerto están en **`config_robot.py`**, que es el único archivo que hay que tocar cuando nos den las definitivas:

```python
LETRAS = {
    "ADELANTE": "F",
    "IZQUIERDA": "I",
    "DERECHA": "D",
    "PARAR": "S",
}
PUERTO_SERIAL = None   # "COM3", "/dev/ttyUSB0"...
BAUDIOS = 9600
TERMINADOR = ""        # "\n" si el robot lee por líneas
```

Con `PUERTO_SERIAL = None` las letras solo se imprimen en consola, que es lo que sirve para probar con los videos. Al poner un puerto se envían por serial (hace falta `pip install pyserial`). La letra solo se manda cuando la acción cambia, para no saturar al robot con la misma letra en cada frame. Al cerrar el programa se envía PARAR.

El resto del código trabaja con los nombres de las acciones (`Accion.ADELANTE`, etc.) y nunca con las letras. Si el robot resulta usar otro protocolo, basta con escribir otra clase de salida con los métodos `escribir()` y `cerrar()` en `comunicacion_robot.py`.

## Cómo está organizado

| Archivo | Qué hace |
|---|---|
| `main.py` | Abre el video o la cámara y conecta todas las piezas en el loop principal. |
| `seguidor_linea.py` | Encuentra la línea y propone ADELANTE / IZQUIERDA / DERECHA. |
| `preprocesador.py` | Máscara de píxeles rojo o verde saturados para buscar las señales. |
| `detector_senales.py` | Filtra por forma los contornos de la máscara (octágonos). |
| `clasificador_color.py` | K-Means sobre el interior del octágono para decidir si es rojo o verde. |
| `control_senales.py` | Máquina de estados: SIGUIENDO / DETENIDO. |
| `comunicacion_robot.py` | Traduce la acción a su letra y la envía. |
| `config_robot.py` | Letras y puerto del robot. |
| `interfaz_control.py` | Ventanas y trackbars. |
| `visualizador.py` | Arma el panel que se ve en pantalla. |

## Seguidor de línea

La cámara ve el frente del robot en la parte de abajo de la imagen, así que la línea se lee en dos franjas horizontales por encima de él: una **cercana** (justo delante del robot) y una **lejana** (lo que viene después).

1. **ROI**: se recortan las dos bandas. Su posición y alto se dan en porcentaje del frame, así funciona igual con cualquier resolución.
2. **Segmentación HSV**: la línea es oscura y poco saturada. Se filtra por valor y saturación máximos, con lo que los carteles rojos y verdes (muy saturados) no se confunden con la línea aunque estén en sombra.
3. **Morfología**: un cierre rellena los huecos del trazo (la línea está pintada a mano y tiene rayas claras). Después viene una apertura con un kernel del grosor mínimo de la línea, que borra todo lo que sea más delgado que la pista, como cables o rayas.
4. **Contornos y centroide**: en cada banda se toma el contorno más grande que la cruce de arriba a abajo, y se calcula su centroide con momentos. La exigencia de cruzar la banda descarta la cinta negra horizontal que sostiene los carteles.
5. **Error respecto al centro del robot**: el punto objetivo es un promedio ponderado de los dos centroides. El lejano hace que el robot empiece a girar antes de llegar a la curva. El error es la distancia entre ese punto y el centro de la imagen.
6. **Continuidad**: la línea no salta de un lado al otro entre una banda y otra ni de un frame al siguiente. Si una detección rompe esa continuidad (una sombra o un pedazo de cinta), se descarta.
7. **Decisión**: si el error está dentro de la tolerancia, ADELANTE; si no, IZQUIERDA o DERECHA según el signo.
8. **Línea perdida**: se repite la última acción que se tomó con la línea a la vista. Si se salió por la derecha, el robot sigue girando a la derecha hasta reencontrarla, que es justo el caso de los videos `uploads/descarriado/`. Si nunca vio la línea, se queda quieto.
9. **Filtro temporal**: una acción nueva solo reemplaza a la actual si se repite 3 frames seguidos, para que un frame con ruido no haga zigzaguear al robot.

## Señales PARE y SIGA

1. **ROI**: se buscan solo en la parte de arriba del frame. Abajo está el robot, que tiene pilas verdes y piezas rojas y amarillas.
2. **Segmentación por color**: blur, conversión a HSV y umbral por matiz (el rojo en los dos extremos del rango H, el verde en el medio) con saturación mínima. Luego apertura y cierre.
3. **Forma**: un contorno se acepta si tiene el área mínima (o sea, el cartel está cerca), no toca el borde de la imagen, su `approxPolyDP` da entre 7 y 9 vértices, su relación de aspecto es cercana a 1 y es sólido (área contra área de la envolvente convexa).
4. **Color con K-Means**: se agrupan en dos clusters los píxeles de adentro del octágono (el fondo y las letras blancas), y el matiz del cluster más grande decide PARE o SIGA.

### Máquina de estados

```
SIGUIENDO --(PARE visto 3 frames seguidos)--> DETENIDO   (envía PARAR)
DETENIDO  --(SIGA visto 3 frames seguidos)--> SIGUIENDO  (vuelve a obedecer a la línea)
```

Cuando el robot reanuda con SIGA, durante un tiempo (trackbar `Espera`) se ignora PARE, para que el mismo cartel no lo vuelva a frenar si sigue a la vista. Si en un frame aparecen varias señales, manda la más grande, que es la más cercana.

### Nota sobre los videos de prueba

Los carteles de los videos de `uploads/ideal/` son cuadrados girados, no octágonos, y approxPolyDP les da 4 o 5 lados. Con los valores por defecto (octágono, 7 a 9 lados) el programa no los toma como señales. Para probar con esos videos hay que bajar el trackbar `Lados Min` a 4. Así probado, en los cuatro videos el robot ve SIGA y sigue, llega a PARE y se detiene, sin falsos positivos por la cartulina roja que aparece a un lado de la pista en `video1`.

## Limitaciones y posibles mejoras

- La segmentación depende de la luz. Con sombras fuertes o reflejos hay que recalibrar los umbrales (`Umbral Val`, `Sat Min`, `Val Min`). Una mejora sería normalizar el brillo de cada banda antes de umbralizar, o usar el canal L de CIELab.
- El control es de tipo encendido/apagado: izquierda, derecha o adelante, sin velocidades intermedias. Si el robot aceptara comandos con intensidad, el mismo error en píxeles serviría para un control proporcional y los giros serían más suaves.
- El centro del robot se asume en la mitad de la imagen. Si la cámara queda corrida, se corrige con el trackbar `Centro`.
- Cuando un cartel tapa la línea justo delante del robot, el seguidor depende de la banda lejana o de la memoria. En tramos largos tapados podría desviarse.
- Un cartel que queda cortado por el límite de la ROI puede dar un número de lados equivocado. Se podría ajustar la ROI o descartar también los contornos que tocan ese límite.
- Las letras y el puerto del robot todavía son provisionales.

## Comparación con otros equipos

_Pendiente: completar después de la socialización en clase._
