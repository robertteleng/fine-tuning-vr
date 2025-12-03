#!/usr/bin/env python3
"""
Auto-anotación de imágenes usando template matching.

Uso:
    python scripts/auto_annotate.py --frames data/frames --template data/template.jpg --output data/labels
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# PASO 1: Cargar template
# =============================================================================
def cargar_template(template_path: str) -> np.ndarray:
    """
    Carga la imagen del template en escala de grises.

    Args:
        template_path: Ruta a la imagen del template

    Returns:
        Template en escala de grises
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"No se pudo cargar el template: {template_path}")
    return template


# =============================================================================
# PASO 2: Template matching a múltiples escalas
# =============================================================================
def match_template_multiscale(
    imagen: np.ndarray,
    template: np.ndarray,
    escalas: list = None,
    umbral: float = 0.65
) -> list:
    """
    Busca el template en la imagen a diferentes escalas.

    ¿Por qué múltiples escalas?
    - La caja puede estar cerca (grande) o lejos (pequeña)
    - El template tiene un tamaño fijo
    - Probamos redimensionar el template para encontrar coincidencias

    Args:
        imagen: Imagen donde buscar (escala de grises)
        template: Template a buscar (escala de grises)
        escalas: Lista de factores de escala (ej: [0.5, 1.0, 1.5])
        umbral: Mínima correlación para considerar una detección (0-1)

    Returns:
        Lista de detecciones: [{'x', 'y', 'w', 'h', 'conf'}, ...]
    """
    if escalas is None:
        # Escalas desde 0.3x hasta 2.0x del template original
        escalas = np.linspace(0.3, 2.0, 20)

    detecciones = []
    h_template, w_template = template.shape[:2]

    for escala in escalas:
        # Redimensionar template
        nuevo_w = int(w_template * escala)
        nuevo_h = int(h_template * escala)

        # Evitar templates más grandes que la imagen
        if nuevo_w >= imagen.shape[1] or nuevo_h >= imagen.shape[0]:
            continue

        # Evitar templates muy pequeños
        if nuevo_w < 10 or nuevo_h < 10:
            continue

        template_escalado = cv2.resize(template, (nuevo_w, nuevo_h))

        # Template matching
        # TM_CCOEFF_NORMED: correlación normalizada, valores entre -1 y 1
        resultado = cv2.matchTemplate(imagen, template_escalado, cv2.TM_CCOEFF_NORMED)

        # Encontrar todas las ubicaciones por encima del umbral
        ubicaciones = np.where(resultado >= umbral)

        for (y, x) in zip(*ubicaciones):
            detecciones.append({
                'x': x,
                'y': y,
                'w': nuevo_w,
                'h': nuevo_h,
                'conf': resultado[y, x]
            })

    return detecciones


# =============================================================================
# PASO 3: Non-Maximum Suppression (NMS)
# =============================================================================
def nms(detecciones: list, iou_umbral: float = 0.3) -> list:
    """
    Elimina detecciones duplicadas usando Non-Maximum Suppression.

    ¿Por qué NMS?
    - El template matching encuentra la misma caja muchas veces
    - A diferentes escalas y posiciones ligeramente distintas
    - NMS mantiene solo la mejor detección de cada grupo

    Algoritmo:
    1. Ordenar por confianza (mayor primero)
    2. Tomar la mejor, eliminar las que se solapan mucho con ella
    3. Repetir hasta procesar todas

    Args:
        detecciones: Lista de detecciones
        iou_umbral: Máximo solapamiento permitido (0-1)

    Returns:
        Lista filtrada de detecciones
    """
    if len(detecciones) == 0:
        return []

    # Convertir a arrays para operaciones vectorizadas
    boxes = np.array([[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']]
                      for d in detecciones])
    scores = np.array([d['conf'] for d in detecciones])

    # Coordenadas de las cajas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Área de cada caja
    areas = (x2 - x1) * (y2 - y1)

    # Ordenar por confianza (índices de mayor a menor)
    orden = scores.argsort()[::-1]

    mantener = []

    while len(orden) > 0:
        # Tomar el índice con mayor confianza
        i = orden[0]
        mantener.append(i)

        # Calcular IoU con el resto
        xx1 = np.maximum(x1[i], x1[orden[1:]])
        yy1 = np.maximum(y1[i], y1[orden[1:]])
        xx2 = np.minimum(x2[i], x2[orden[1:]])
        yy2 = np.minimum(y2[i], y2[orden[1:]])

        # Área de intersección
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        interseccion = w * h

        # IoU = intersección / unión
        iou = interseccion / (areas[i] + areas[orden[1:]] - interseccion)

        # Mantener solo las cajas con IoU menor al umbral
        indices_mantener = np.where(iou <= iou_umbral)[0]
        orden = orden[indices_mantener + 1]

    return [detecciones[i] for i in mantener]


# =============================================================================
# PASO 4: Convertir a formato YOLO
# =============================================================================
def a_formato_yolo(detecciones: list, ancho_img: int, alto_img: int, clase: int = 0) -> list:
    """
    Convierte detecciones a formato YOLO normalizado.

    Formato YOLO: clase x_centro y_centro ancho alto
    - Todos los valores normalizados entre 0 y 1
    - x_centro, y_centro: centro de la caja
    - ancho, alto: tamaño de la caja

    Args:
        detecciones: Lista de detecciones con x, y, w, h en píxeles
        ancho_img: Ancho de la imagen en píxeles
        alto_img: Alto de la imagen en píxeles
        clase: Índice de la clase (default 0)

    Returns:
        Lista de strings en formato YOLO
    """
    lineas = []

    for det in detecciones:
        # Calcular centro
        x_centro = (det['x'] + det['w'] / 2) / ancho_img
        y_centro = (det['y'] + det['h'] / 2) / alto_img

        # Normalizar dimensiones
        ancho = det['w'] / ancho_img
        alto = det['h'] / alto_img

        # Formato: clase x_centro y_centro ancho alto
        linea = f"{clase} {x_centro:.6f} {y_centro:.6f} {ancho:.6f} {alto:.6f}"
        lineas.append(linea)

    return lineas


# =============================================================================
# PASO 5: Procesar una imagen
# =============================================================================
def procesar_imagen(
    imagen_path: str,
    template: np.ndarray,
    umbral: float = 0.65,
    iou_umbral: float = 0.3
) -> list:
    """
    Procesa una imagen completa: detecta y filtra.

    Args:
        imagen_path: Ruta a la imagen
        template: Template en escala de grises
        umbral: Umbral de detección
        iou_umbral: Umbral de NMS

    Returns:
        Lista de líneas en formato YOLO
    """
    # Cargar imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        print(f"  Error: No se pudo cargar {imagen_path}")
        return []

    alto, ancho = imagen.shape[:2]

    # Detectar
    detecciones = match_template_multiscale(imagen, template, umbral=umbral)

    # Filtrar duplicados
    detecciones_filtradas = nms(detecciones, iou_umbral=iou_umbral)

    # Convertir a YOLO
    lineas_yolo = a_formato_yolo(detecciones_filtradas, ancho, alto)

    return lineas_yolo


# =============================================================================
# PASO 6: Main
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto-anotación con template matching')
    parser.add_argument('--frames', type=str, required=True, help='Carpeta con imágenes')
    parser.add_argument('--template', type=str, required=True, help='Imagen del template')
    parser.add_argument('--output', type=str, required=True, help='Carpeta de salida para labels')
    parser.add_argument('--umbral', type=float, default=0.65, help='Umbral de detección (0-1)')
    parser.add_argument('--iou', type=float, default=0.3, help='Umbral IoU para NMS (0-1)')

    args = parser.parse_args()

    # Crear carpeta de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar template
    print(f"Cargando template: {args.template}")
    template = cargar_template(args.template)
    print(f"  Tamaño: {template.shape[1]}x{template.shape[0]} px")

    # Buscar imágenes
    frames_dir = Path(args.frames)
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imagenes = []
    for ext in extensiones:
        imagenes.extend(frames_dir.glob(ext))
    imagenes = sorted(imagenes)

    print(f"Imágenes encontradas: {len(imagenes)}")

    # Procesar cada imagen
    total_detecciones = 0

    for img_path in tqdm(imagenes, desc="Procesando"):
        lineas = procesar_imagen(str(img_path), template, args.umbral, args.iou)

        # Guardar archivo .txt
        label_path = output_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(lineas))

        total_detecciones += len(lineas)

    # Resumen
    print(f"\n{'='*50}")
    print(f"RESUMEN")
    print(f"{'='*50}")
    print(f"Imágenes procesadas: {len(imagenes)}")
    print(f"Total detecciones: {total_detecciones}")
    print(f"Promedio por imagen: {total_detecciones/max(len(imagenes),1):.1f}")
    print(f"Labels guardados en: {output_dir}")


if __name__ == '__main__':
    main()
