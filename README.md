# Fine-Tuning YOLOv8/v11 para Detección de Cajas VR

## Objetivo del Proyecto

Este proyecto implementa un modelo de detección de objetos especializado en identificar **cajas de experiencias de Realidad Virtual (VR)** en capturas de pantalla. El modelo está optimizado para detectar dos clases específicas de cajas que aparecen en la interfaz de una aplicación VR.

### ¿Por qué Fine-Tuning en lugar de Zero-Shot?

| Aspecto | Zero-Shot (Ej: CLIP, Grounding DINO) | Fine-Tuning (YOLOv8/v11) |
|---------|--------------------------------------|--------------------------|
| **Clases** | Abiertas, cualquier texto | Fijas, predefinidas |
| **Precisión** | ~70-85% en objetos comunes | >95% en clases específicas |
| **Velocidad** | Más lento (modelos grandes) | Muy rápido (~2-5ms/imagen) |
| **Consistencia** | Variable según el prompt | Altamente consistente |
| **Uso en producción** | Requiere más recursos | Ligero y eficiente |

**Decisión**: Para nuestro caso de uso con **exactamente 2 clases fijas** de cajas VR, el fine-tuning ofrece:
- ✅ Máxima precisión posible
- ✅ Inferencia ultra-rápida
- ✅ Modelo pequeño (~6MB con YOLOv8n)
- ✅ Funcionamiento offline
- ✅ Resultados reproducibles

---

## Especificaciones de Hardware

### Configuración Actual
| Componente | Especificación |
|------------|----------------|
| GPU | NVIDIA RTX 2060 (6GB VRAM) |
| Limitaciones | Batch size máximo: 8-16 |
| Tiempo estimado/época | ~2-5 min (depende del dataset) |

### Configuración Futura (en ~2 semanas)
| Componente | Especificación |
|------------|----------------|
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM esperada) |
| Mejoras esperadas | Batch size: 32-64, ~2-3x más rápido |

> **Nota**: Ver [docs/benchmarks.md](docs/benchmarks.md) para comparativa de rendimiento.

---

## Estructura del Proyecto

```
fine_tuning_cajas_vr/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
├── config.yaml                  # Configuración de entrenamiento
├── data/                        # Dataset (exportado de Roboflow)
│   ├── train/
│   │   ├── images/             # Imágenes de entrenamiento
│   │   └── labels/             # Anotaciones YOLO format (.txt)
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── scripts/
│   ├── train.py                # Script de entrenamiento
│   └── inference.py            # Script de inferencia
├── notebooks/
│   └── 01_data_exploration.ipynb  # Exploración del dataset
├── docs/
│   └── benchmarks.md           # Métricas y comparativas
├── models/                     # Modelos entrenados guardados
└── runs/                       # Logs de entrenamiento (TensorBoard)
```

---

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar instalación de CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Guía Paso a Paso

### 1. Captura de Screenshots VR

Para crear un dataset de calidad, sigue estas recomendaciones:

#### Método de Captura
```
1. Abre la aplicación VR en modo escritorio (si es posible)
2. Usa herramientas de captura:
   - Windows: Win + Shift + S (Snipping Tool)
   - OBS Studio: Para capturas consistentes
   - NVIDIA ShadowPlay: Alt + F1

3. Captura variedad de escenas:
   - Diferentes posiciones de las cajas
   - Diferentes fondos/contextos
   - Diferentes tamaños de caja en pantalla
   - Con y sin oclusiones parciales
```

#### Recomendaciones de Cantidad
| Fase | Imágenes por clase | Total mínimo |
|------|-------------------|--------------|
| MVP/Prueba | 50-100 | 100-200 |
| Producción básica | 200-500 | 400-1000 |
| Alta precisión | 500-1000+ | 1000-2000+ |

#### Tips para Mejores Resultados
- ✅ Incluir variaciones de iluminación
- ✅ Capturar cajas en diferentes escalas
- ✅ Incluir casos "difíciles" (parcialmente visibles)
- ✅ Mantener resolución consistente (ej: 1920x1080)
- ❌ Evitar imágenes borrosas o de baja calidad
- ❌ Evitar demasiadas imágenes casi idénticas

---

### 2. Anotación con Roboflow

#### Crear Proyecto en Roboflow

1. **Ir a [roboflow.com](https://roboflow.com)** y crear cuenta gratuita

2. **Crear nuevo proyecto**:
   - Project Type: `Object Detection`
   - Project Name: `vr-boxes-detection`
   - License: Tu preferencia
   - Annotation Group: `vr_boxes`

3. **Definir clases** (ajusta según tus cajas específicas):
   ```
   Clase 0: vr_box_type_a    # Ejemplo: Caja de experiencia principal
   Clase 1: vr_box_type_b    # Ejemplo: Caja de menú/navegación
   ```

#### Proceso de Anotación

```
1. Subir imágenes:
   - Drag & drop las capturas
   - Roboflow detectará duplicados automáticamente

2. Anotar cada imagen:
   - Usar herramienta de bounding box
   - Dibujar rectángulo alrededor de cada caja
   - Asignar la clase correcta
   - Asegurar que el box cubra toda la caja

3. Tips de anotación:
   - Ser consistente en los bordes
   - No dejar espacio extra innecesario
   - Anotar TODAS las instancias visibles
   - Marcar como "null" imágenes sin objetos
```

#### Generar Dataset

```
1. Ir a "Generate" en el proyecto

2. Configurar preprocessing:
   - Auto-Orient: ON
   - Resize: 640x640 (estándar YOLO)

3. Configurar augmentation (OPCIONAL para datasets pequeños):
   - Flip: Horizontal
   - Rotation: ±15°
   - Brightness: ±15%
   - (No exceder, YOLO hace augmentation interno)

4. Split del dataset:
   - Train: 70%
   - Valid: 20%
   - Test: 10%

5. Generar nueva versión
```

---

### 3. Exportar y Colocar Archivos

#### Exportar desde Roboflow

```
1. En tu dataset generado, click "Export"

2. Seleccionar formato:
   ⭐ "YOLOv8" (formato nativo, recomendado)

3. Método de descarga:
   - "Download zip" para descarga manual
   - O usar código Python (más conveniente):
```

```python
# Código de exportación (Roboflow te lo genera)
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("tu-workspace").project("vr-boxes-detection")
dataset = project.version(1).download("yolov8")
```

#### Estructura Esperada del Export

```
vr-boxes-detection-1/
├── data.yaml           # Configuración del dataset
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt  # Formato: class x_center y_center width height
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### Colocar en el Proyecto

```bash
# Opción 1: Copiar contenido a carpeta data/
cp -r vr-boxes-detection-1/train/* fine_tuning_cajas_vr/data/train/
cp -r vr-boxes-detection-1/valid/* fine_tuning_cajas_vr/data/valid/
cp -r vr-boxes-detection-1/test/* fine_tuning_cajas_vr/data/test/

# Opción 2: Modificar config.yaml para apuntar a la carpeta descargada
# (ver config.yaml para instrucciones)
```

---

### 4. Entrenamiento

```bash
# Entrenar el modelo
cd fine_tuning_cajas_vr
python scripts/train.py

# Monitorear con TensorBoard (opcional, en otra terminal)
tensorboard --logdir runs/
```

---

### 5. Inferencia

```bash
# Probar en una imagen
python scripts/inference.py --source path/to/image.jpg

# Probar en un directorio
python scripts/inference.py --source path/to/images/

# Probar en video
python scripts/inference.py --source path/to/video.mp4
```

---

## Checklist de Upgrade GPU (RTX 2060 → RTX 5060 Ti)

Cuando actualices a la RTX 5060 Ti, sigue estos pasos:

### Parámetros a Ajustar

| Parámetro | RTX 2060 (6GB) | RTX 5060 Ti (16GB) | Archivo |
|-----------|----------------|-------------------|---------|
| `batch` | 8 | 32 (o 64) | config.yaml |
| `workers` | 4 | 8-12 | config.yaml |
| `imgsz` | 640 | 640 (o 1280) | config.yaml |
| `cache` | False/ram | ram | config.yaml |

### Procedimiento de Upgrade

```bash
# 1. Actualizar drivers NVIDIA
# Descargar de: https://www.nvidia.com/drivers

# 2. Verificar nueva GPU
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. Actualizar config.yaml
# Cambiar batch: 8 → batch: 32
# Cambiar workers: 4 → workers: 8

# 4. (Opcional) Re-instalar PyTorch para CUDA más reciente
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu124

# 5. Ejecutar benchmark
python scripts/train.py --epochs 1  # Prueba rápida

# 6. Documentar resultados en docs/benchmarks.md
```

### Verificación Post-Upgrade

```python
# Script de verificación
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Decisiones Técnicas

### Modelo Base: YOLOv8n vs YOLOv11n

| Aspecto | YOLOv8n | YOLOv11n |
|---------|---------|----------|
| Madurez | Muy estable, amplia documentación | Más reciente, mejoras incrementales |
| Velocidad | ~2ms/imagen | ~1.8ms/imagen |
| Precisión | Excelente | Ligeramente mejor |
| Tamaño | 6.3 MB | 5.4 MB |
| **Recomendación** | ✅ Para producción estable | ✅ Para mejor rendimiento |

**Decisión actual**: Usar `yolov8n.pt` por estabilidad. Cambiar a `yolov11n.pt` es trivial si se desea.

### Hiperparámetros Elegidos

Ver [config.yaml](config.yaml) para todos los parámetros con explicaciones detalladas.

---

## Resultados

> **TODO**: Completar después del primer entrenamiento

| Métrica | Valor |
|---------|-------|
| mAP@50 | - |
| mAP@50-95 | - |
| Precisión | - |
| Recall | - |
| Tiempo/imagen | - |

Ver [docs/benchmarks.md](docs/benchmarks.md) para historial completo.

---

## Troubleshooting

### Error: CUDA out of memory
```bash
# Reducir batch size en config.yaml
batch: 4  # En lugar de 8
```

### Error: No module named 'ultralytics'
```bash
pip install ultralytics
```

### Entrenamiento muy lento
```bash
# Verificar que CUDA está activo
python -c "import torch; print(torch.cuda.is_available())"  # Debe ser True

# Si es False, reinstalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Las predicciones son malas
1. Verificar calidad de anotaciones en notebook de exploración
2. Aumentar número de épocas
3. Agregar más datos de entrenamiento
4. Verificar que no hay data leakage (imágenes repetidas en train/valid)

---

## Referencias

- [Documentación Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Roboflow - Guía de Anotación](https://docs.roboflow.com/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
- [Tips de Fine-Tuning](https://docs.ultralytics.com/guides/model-training-tips/)

---

## Licencia

Este proyecto es para uso interno/educativo. El modelo base YOLOv8 está bajo licencia AGPL-3.0.

---

*Última actualización: Diciembre 2024*
