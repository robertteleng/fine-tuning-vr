# Diario de Desarrollo - YOLO VR Box Detection

## Resumen del Proyecto
Fine-tuning de YOLO para detectar cajas con patrón tablero amarillo-negro en VR para navegación asistida.

---

## 2024-12-03 - Inicio del Desarrollo

### Estado Inicial del Proyecto
**Archivos existentes:**
- `config.yaml` - Configuración de entrenamiento
- `requirements.txt` - Dependencias
- `scripts/train.py` - Script de entrenamiento
- `scripts/inference.py` - Script de inferencia
- `docs/DOCUMENTATION.md` - Documentación técnica
- `docs/START_PROMPT.md` - Especificación del proyecto
- `data/` - Dataset con train/valid/test splits, template y video

**Scripts pendientes de crear:**
1. `scripts/auto_annotate.py` - Auto-anotación con template matching
2. `scripts/split_dataset.py` - División 80/20 del dataset
3. `scripts/evaluate.py` - Evaluación de métricas
4. `scripts/visualize_annotations.py` - Verificar anotaciones
5. `scripts/export_tensorrt.py` - Exportación TensorRT
6. `scripts/benchmark.py` - Comparación de velocidad
7. `data/cajas.yaml` - Configuración del dataset

---

## Hitos

### Hito 1: [PENDIENTE] auto_annotate.py
- **Estado:** En progreso
- **Descripción:** Script de auto-anotación con template matching multi-escala y NMS
- **Testing:**
- **Commit:**

---

### Hito 2: [PENDIENTE] split_dataset.py
- **Estado:** Pendiente
- **Descripción:** División reproducible del dataset 80/20
- **Testing:**
- **Commit:**

---

### Hito 3: [PENDIENTE] evaluate.py
- **Estado:** Pendiente
- **Descripción:** Evaluación de métricas mAP, precision, recall, F1
- **Testing:**
- **Commit:**

---

### Hito 4: [PENDIENTE] visualize_annotations.py
- **Estado:** Pendiente
- **Descripción:** Visualización de anotaciones para verificación
- **Testing:**
- **Commit:**

---

### Hito 5: [PENDIENTE] export_tensorrt.py
- **Estado:** Pendiente
- **Descripción:** Exportación a TensorRT con FP16
- **Testing:**
- **Commit:**

---

### Hito 6: [PENDIENTE] benchmark.py
- **Estado:** Pendiente
- **Descripción:** Benchmark de formatos (PyTorch, TensorRT FP32, TensorRT FP16)
- **Testing:**
- **Commit:**

---

### Hito 7: [PENDIENTE] data/cajas.yaml
- **Estado:** Pendiente
- **Descripción:** Configuración del dataset para YOLO
- **Testing:**
- **Commit:**

---

## Notas y Decisiones

- Flujo de trabajo: implementar → testing → documentar → commit
- Hardware objetivo: RTX 2060 (6GB) → RTX 5060 Ti (16GB)
- Modelo: YOLOv12n para velocidad, posible upgrade a YOLOv12s

---
