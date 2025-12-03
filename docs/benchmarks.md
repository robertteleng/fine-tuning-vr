# Benchmarks y Métricas de Entrenamiento

Este documento registra las métricas de entrenamiento y comparativas de rendimiento entre diferentes configuraciones de hardware y modelos.

---

## Tabla de Contenidos

1. [Métricas de Referencia](#métricas-de-referencia)
2. [Historial de Entrenamientos](#historial-de-entrenamientos)
3. [Comparativa RTX 2060 vs RTX 5060 Ti](#comparativa-rtx-2060-vs-rtx-5060-ti)
4. [Mejores Configuraciones](#mejores-configuraciones)

---

## Métricas de Referencia

### Métricas YOLO Explicadas

| Métrica | Descripción | Valor Ideal |
|---------|-------------|-------------|
| **mAP@50** | Mean Average Precision con IoU 0.5 | > 0.90 |
| **mAP@50-95** | mAP promediando IoUs de 0.5 a 0.95 | > 0.70 |
| **Precision** | Detecciones correctas / Total detecciones | > 0.90 |
| **Recall** | Detecciones correctas / Total objetos reales | > 0.85 |
| **F1-Score** | Media armónica de Precision y Recall | > 0.87 |

### Velocidad de Inferencia

| Métrica | RTX 2060 Target | RTX 5060 Ti Target |
|---------|-----------------|-------------------|
| Tiempo/imagen | < 10ms | < 5ms |
| FPS | > 100 | > 200 |

---

## Historial de Entrenamientos

### Plantilla para Registrar Entrenamiento

```
## Entrenamiento: [FECHA] - [NOMBRE_EXPERIMENTO]

### Configuración
- **Modelo base**: yolov8n.pt / yolov11n.pt
- **GPU**: RTX 2060 / RTX 5060 Ti
- **Batch size**: X
- **Image size**: 640 / 1280
- **Épocas**: X (early stop en Y)
- **Dataset**: X train / Y valid / Z test

### Métricas Finales
| Clase | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| vr_box_type_a | - | - | - | - |
| vr_box_type_b | - | - | - | - |
| **Total** | - | - | - | - |

### Tiempos
- Tiempo total: Xh Ym
- Tiempo por época: ~Xm
- Inferencia: Xms/imagen

### Observaciones
- [Notas sobre el entrenamiento]
- [Problemas encontrados]
- [Mejoras para siguiente iteración]
```

---

## Entrenamientos Registrados

### Entrenamiento: [PENDIENTE] - Baseline RTX 2060

> **TODO**: Completar después del primer entrenamiento

#### Configuración
- **Modelo base**: yolov8n.pt
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **Batch size**: 8
- **Image size**: 640
- **Épocas**: 100 (patience: 20)
- **Dataset**: - train / - valid / - test

#### Métricas Finales
| Clase | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| vr_box_type_a | - | - | - | - |
| vr_box_type_b | - | - | - | - |
| **Total** | - | - | - | - |

#### Tiempos
- Tiempo total: -
- Tiempo por época: -
- Inferencia: -

#### Observaciones
- Primer entrenamiento de baseline
- [Agregar observaciones post-entrenamiento]

---

### Entrenamiento: [PENDIENTE] - Post-Upgrade RTX 5060 Ti

> **TODO**: Completar después de recibir la RTX 5060 Ti

#### Configuración
- **Modelo base**: yolov8n.pt / yolov11n.pt
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **Batch size**: 32
- **Image size**: 640
- **Épocas**: 100
- **Dataset**: - train / - valid / - test

#### Métricas Finales
| Clase | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| vr_box_type_a | - | - | - | - |
| vr_box_type_b | - | - | - | - |
| **Total** | - | - | - | - |

#### Tiempos
- Tiempo total: -
- Tiempo por época: -
- Inferencia: -

#### Observaciones
- Comparar velocidad vs RTX 2060
- [Agregar observaciones post-entrenamiento]

---

## Comparativa RTX 2060 vs RTX 5060 Ti

### Especificaciones de Hardware

| Especificación | RTX 2060 | RTX 5060 Ti (Esperado) |
|----------------|----------|------------------------|
| VRAM | 6GB GDDR6 | 16GB GDDR7 (estimado) |
| CUDA Cores | 1920 | ~5000+ (estimado) |
| Tensor Cores | Gen 2 | Gen 5 (estimado) |
| Arquitectura | Turing | Blackwell |
| TDP | 160W | ~180W (estimado) |

### Comparativa de Entrenamiento

| Parámetro | RTX 2060 | RTX 5060 Ti |
|-----------|----------|-------------|
| Batch size máximo | 8-16 | 32-64 |
| Imgsz máximo | 640 | 640-1280 |
| Workers óptimos | 4 | 8-12 |
| Cache | False/disco | RAM |
| Tiempo/época (estimado) | ~3-5 min | ~1-2 min |
| **Speedup esperado** | 1x | ~2-3x |

### Gráficos de Comparación

> **TODO**: Agregar gráficos después de tener datos de ambas GPUs

```
Tiempo por época (minutos)
RTX 2060:    ████████████████████ 5.0 min
RTX 5060 Ti: ████████ 2.0 min (estimado)

Batch size
RTX 2060:    ████ 8
RTX 5060 Ti: ████████████████ 32 (estimado)
```

---

## Mejores Configuraciones

### Para RTX 2060 (6GB VRAM)

```yaml
# config.yaml optimizado para RTX 2060
model: yolov8n.pt
batch: 8
imgsz: 640
workers: 4
cache: False
amp: True  # Importante para ahorrar memoria
epochs: 100
patience: 20
```

**Tips específicos:**
- No usar cache en RAM (memoria limitada)
- Mantener batch=8 para evitar OOM
- AMP (mixed precision) es crucial
- Cerrar otras aplicaciones que usen GPU

### Para RTX 5060 Ti (16GB VRAM) - Estimado

```yaml
# config.yaml optimizado para RTX 5060 Ti
model: yolov8n.pt  # o yolov11n.pt
batch: 32
imgsz: 640  # o 1280 si necesitas más detalle
workers: 8
cache: ram  # Si tienes >=32GB de RAM del sistema
amp: True
epochs: 100
patience: 20
```

**Tips específicos:**
- Aprovechar batch grande para mejor convergencia
- Considerar imgsz=1280 para objetos pequeños
- Cache en RAM acelera significativamente
- Posibilidad de usar modelos más grandes (yolov8s, yolov8m)

---

## Experimentos Futuros

### A probar con RTX 5060 Ti

- [ ] Comparar YOLOv8n vs YOLOv11n
- [ ] Probar batch=64 si VRAM lo permite
- [ ] Evaluar imgsz=1280 vs 640
- [ ] Test con modelo YOLOv8s (más grande)
- [ ] Benchmark de inferencia con TensorRT

### Optimizaciones pendientes

- [ ] Exportar a ONNX para producción
- [ ] Probar quantización INT8
- [ ] Evaluar TensorRT para máxima velocidad
- [ ] Test en diferentes resoluciones de entrada

---

## Notas de Implementación

### Cómo agregar un nuevo benchmark

1. Ejecutar entrenamiento:
   ```bash
   python scripts/train.py --name mi_experimento
   ```

2. Copiar métricas desde `runs/train/mi_experimento/results.csv`

3. Agregar nueva sección usando la plantilla de arriba

4. Actualizar gráficos comparativos si aplica

### Archivos relacionados

- Métricas detalladas: `runs/train/[experimento]/results.csv`
- Curvas de entrenamiento: `runs/train/[experimento]/results.png`
- Mejor modelo: `runs/train/[experimento]/weights/best.pt`
- TensorBoard: `tensorboard --logdir runs/train`

---

## Referencias

- [Ultralytics YOLOv8 Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [mAP Explained](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Tips de Optimización GPU](https://docs.ultralytics.com/guides/model-training-tips/)

---

*Última actualización: Diciembre 2024*
