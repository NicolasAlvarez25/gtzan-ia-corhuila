# Clasificación de Géneros Musicales con CNN (GTZAN)

Este repositorio contiene el proyecto final de la asignatura de **Inteligencia Artificial** de la **Corporación Universitaria del Huila, CORHUILA**.

El proyecto implementa un sistema basado en una **Red Neuronal Convolucional (CNN)** capaz de clasificar fragmentos de audio en **nueve géneros musicales**, utilizando espectrogramas Mel de tres segundos representados como imágenes de 128 × 128 píxeles en escala de grises.

El sistema incluye:

- Entrenamiento de una CNN con TensorFlow y Keras.
- Procesamiento de espectrogramas Mel.
- Evaluación automática del modelo.
- Generación de métricas, reportes y gráficas.
- Aplicación web desarrollada con Flask.
- Clasificación de archivos de audio en formato `.wav`.
- Visualización de resultados desde la aplicación web.

---

## 👥 Equipo de trabajo

- **Nicolás Gabriel Álvarez Aguirre**
- **Sergio Luis Angel Romero**

---

## 🎯 Objetivo del proyecto

Desarrollar un sistema de inteligencia artificial capaz de identificar el género musical de un fragmento de audio, utilizando espectrogramas Mel y una arquitectura de Red Neuronal Convolucional.

El proyecto aplica conceptos de:

- Aprendizaje profundo.
- Redes neuronales convolucionales.
- Procesamiento digital de audio.
- Clasificación multiclase.
- Regularización de modelos.
- Optimización de hiperparámetros.
- Evaluación mediante métricas de clasificación.
- Integración de modelos de inteligencia artificial en aplicaciones web.

---

## 📊 Dataset y clases

Se utilizó el dataset **3-Second Melspectrograms for GTZAN**, obtenido de Kaggle y basado en el conjunto de datos GTZAN para clasificación de géneros musicales.

Se procesaron **8.100 imágenes de espectrogramas Mel**, distribuidas de la siguiente manera:

| Conjunto | Imágenes |
|---|---:|
| Entrenamiento | 7.200 |
| Validación | 900 |
| Total | 8.100 |

El conjunto se encuentra balanceado:

- 800 imágenes de entrenamiento por género.
- 100 imágenes de validación por género.
- 900 imágenes totales por género.

### Géneros musicales

El modelo clasifica los siguientes nueve géneros:

1. Blues
2. Classical
3. Country
4. Disco
5. Hiphop
6. Metal
7. Pop
8. Reggae
9. Rock

> **Nota técnica:** El género Jazz no fue incluido en esta versión debido a la disponibilidad y organización del dataset utilizado.

---

## 🎵 Representación de los audios

Los archivos de audio son representados mediante **espectrogramas Mel**.

Un espectrograma permite visualizar cómo cambia la energía de las frecuencias de una señal de audio a lo largo del tiempo. La escala Mel aproxima la percepción auditiva humana, por lo que resulta adecuada para tareas de clasificación musical.

Cada entrada del modelo tiene las siguientes características:

| Característica | Valor |
|---|---|
| Representación | Espectrograma Mel |
| Duración | 3 segundos |
| Resolución | 128 × 128 píxeles |
| Canales | 1 |
| Color | Escala de grises |
| Normalización | Valores entre 0 y 1 |

---

## 🧠 Arquitectura optimizada de la CNN

La versión actual utiliza una CNN entrenada desde cero y optimizada para trabajar con espectrogramas Mel.

La arquitectura contiene:

- Cuatro bloques convolucionales.
- Dos capas `Conv2D` por bloque.
- Capas `BatchNormalization`.
- Funciones de activación ReLU.
- Capas `MaxPooling2D`.
- Regularización con `SpatialDropout2D`.
- Capa `GlobalAveragePooling2D`.
- Capa densa de 256 neuronas.
- Capa `Dropout`.
- Capa de salida Softmax con nueve neuronas.

### Resumen técnico

| Elemento | Configuración |
|---|---|
| Entrada | 128 × 128 × 1 |
| Bloques convolucionales | 4 |
| Convoluciones por bloque | 2 |
| Filtros | 32, 64, 128 y 256 |
| Clasificador | GlobalAveragePooling2D y Dense |
| Neuronas en capa densa | 256 |
| Función de activación | ReLU |
| Función de salida | Softmax |
| Clases de salida | 9 |
| Parámetros totales | 1.243.433 |
| Parámetros entrenables | 1.241.001 |
| Parámetros no entrenables | 2.432 |

La capa `Flatten` utilizada anteriormente fue reemplazada por `GlobalAveragePooling2D`. Este cambio redujo la complejidad del clasificador y ayudó a mejorar la capacidad de generalización.

---

## ⚙️ Mejoras implementadas

La primera versión del modelo alcanzaba una exactitud del 75 %. Para mejorar su rendimiento se implementaron los siguientes cambios.

### Batch Normalization

Se agregó `BatchNormalization` después de las capas convolucionales y antes de las funciones de activación.

Esta técnica ayuda a:

- Estabilizar las activaciones.
- Acelerar el entrenamiento.
- Facilitar la convergencia.
- Reducir la sensibilidad a la inicialización de los pesos.

### Spatial Dropout

Se incorporó `SpatialDropout2D` en los bloques convolucionales.

A diferencia del Dropout tradicional, esta técnica desactiva mapas de características completos durante el entrenamiento, reduciendo la dependencia del modelo respecto a características específicas.

### Global Average Pooling

La capa `Flatten` fue reemplazada por `GlobalAveragePooling2D`.

Esto redujo la cantidad de parámetros del clasificador y ayudó a disminuir el sobreajuste.

### Optimizador AdamW

El modelo utiliza el optimizador `AdamW`, que incorpora regularización mediante decaimiento de pesos.

Configuración inicial:

```python
learning_rate = 0.001
weight_decay = 0.0001
```

### ReduceLROnPlateau

Se implementó `ReduceLROnPlateau` para disminuir automáticamente la tasa de aprendizaje cuando la pérdida de validación deja de mejorar.

Durante el entrenamiento, el learning rate se redujo progresivamente desde:

```text
0.001
```

hasta valores cercanos a:

```text
0.00000390625
```

Esto permitió realizar ajustes cada vez más pequeños sobre los pesos del modelo durante las últimas épocas.

### Early Stopping

Se utilizó `EarlyStopping` para detener el entrenamiento cuando la pérdida de validación dejó de mejorar.

El entrenamiento estaba configurado para un máximo de 80 épocas y se detuvo automáticamente en la época 79.

### Model Checkpoint

Se implementó `ModelCheckpoint` para guardar únicamente el modelo con mayor exactitud de validación.

El mejor modelo guardado alcanzó su mayor `val_accuracy` en la época 63.

### Data Augmentation

El aumento de datos incluye:

- Pequeñas traslaciones horizontales.
- Variaciones controladas de contraste.

No se utilizan volteos horizontales o verticales porque podrían modificar el significado temporal o frecuencial de los espectrogramas.

### Reproducibilidad

Se estableció una semilla global con valor `42` para Python, NumPy y TensorFlow.

Esto ayuda a mantener condiciones similares entre diferentes ejecuciones del entrenamiento.

---

## 🖥️ Entorno de entrenamiento

El entrenamiento se ejecutó en **Google Colab** utilizando aceleración por hardware.

| Recurso | Configuración |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Framework | TensorFlow 2.21.0 |
| API de redes neuronales | Keras |
| Entorno | Google Colab |
| Sistema de evaluación | Scikit-learn |
| Generación de gráficas | Matplotlib y Seaborn |

---

## 🔧 Configuración del entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas máximas | 80 |
| Épocas ejecutadas | 79 |
| Mejor época por `val_accuracy` | 63 |
| Batch size | 32 |
| Learning rate inicial | 0.001 |
| Learning rate mínimo configurado | 0.000001 |
| Optimizador | AdamW |
| Weight decay | 0.0001 |
| Función de pérdida | Sparse Categorical Crossentropy |
| Métrica principal | Accuracy |
| Semilla | 42 |
| GPU | NVIDIA Tesla T4 |

---

## ✅ Resultados finales

El modelo optimizado fue evaluado sobre **900 imágenes de validación**, con 100 imágenes por género.

| Métrica | Resultado |
|---|---:|
| Accuracy | **90,56 %** |
| Loss | **0,3213** |
| Precision macro | **0,9102** |
| Recall macro | **0,9056** |
| F1-score macro | **0,9064** |
| Imágenes evaluadas | 900 |
| Mejor época por accuracy | 63 |

---

## 📈 Resultados por género

| Género | Precision | Recall | F1-score | Muestras |
|---|---:|---:|---:|---:|
| Blues | 0,9694 | 0,9500 | 0,9596 | 100 |
| Classical | 0,9901 | 1,0000 | 0,9950 | 100 |
| Country | 0,9545 | 0,8400 | 0,8936 | 100 |
| Disco | 0,8230 | 0,9300 | 0,8732 | 100 |
| Hiphop | 0,8812 | 0,8900 | 0,8856 | 100 |
| Metal | 0,9588 | 0,9300 | 0,9442 | 100 |
| Pop | 0,8558 | 0,8900 | 0,8725 | 100 |
| Reggae | 0,9773 | 0,8600 | 0,9149 | 100 |
| Rock | 0,7818 | 0,8600 | 0,8190 | 100 |

La clase con mejor resultado fue **Classical**, con un F1-score de 0,9950.

La clase con mayor dificultad fue **Rock**, con un F1-score de 0,8190. Esto puede relacionarse con similitudes acústicas entre Rock y otros géneros como Metal, Blues y Country.

---

## 📊 Comparación entre versiones

| Métrica | Modelo anterior | Modelo optimizado | Mejora |
|---|---:|---:|---:|
| Accuracy | 75,00 % | **90,56 %** | **+15,56 puntos porcentuales** |
| Precision macro | 0,7610 | **0,9102** | +0,1492 |
| Recall macro | 0,7500 | **0,9056** | +0,1556 |
| F1-score macro | 0,7490 | **0,9064** | +0,1574 |

La optimización incrementó la exactitud del modelo en **15,56 puntos porcentuales**.

---

## 💾 Archivos generados

Después del entrenamiento se generan automáticamente los siguientes archivos.

### Modelo y clases

```text
web_app/model/modelo_final.keras
web_app/model/clases.json
```

### Resultados y evidencias

```text
web_app/static/accuracy_curve.png
web_app/static/loss_curve.png
web_app/static/training_plot.png
web_app/static/confusion_matrix.png
web_app/static/per_class_metrics.png
web_app/static/evaluation_results.json
web_app/static/classification_report.txt
web_app/static/training_history.json
web_app/static/training_log.csv
```

### Descripción de los archivos

| Archivo | Contenido |
|---|---|
| `modelo_final.keras` | Mejor modelo guardado durante el entrenamiento |
| `clases.json` | Relación entre índices numéricos y géneros |
| `accuracy_curve.png` | Accuracy de entrenamiento y validación |
| `loss_curve.png` | Loss de entrenamiento y validación |
| `training_plot.png` | Curvas combinadas de accuracy y loss |
| `confusion_matrix.png` | Matriz de confusión |
| `per_class_metrics.png` | Precision, recall y F1 por género |
| `evaluation_results.json` | Métricas y resultados en formato JSON |
| `classification_report.txt` | Reporte de clasificación de Scikit-learn |
| `training_history.json` | Historial numérico de las épocas |
| `training_log.csv` | Registro tabular del entrenamiento |

La aplicación web permite consultar los resultados desde la ruta `/resultados`.

---

## 📁 Estructura del proyecto

```text
gtzan-ia-corhuila/
│
├── dataset/
│   └── procesado/
│       ├── train/
│       │   ├── blues/
│       │   ├── classical/
│       │   ├── country/
│       │   ├── disco/
│       │   ├── hiphop/
│       │   ├── metal/
│       │   ├── pop/
│       │   ├── reggae/
│       │   └── rock/
│       │
│       └── val/
│           ├── blues/
│           ├── classical/
│           ├── country/
│           ├── disco/
│           ├── hiphop/
│           ├── metal/
│           ├── pop/
│           ├── reggae/
│           └── rock/
│
├── web_app/
│   ├── model/
│   │   ├── modelo_final.keras
│   │   └── clases.json
│   │
│   ├── static/
│   │   ├── accuracy_curve.png
│   │   ├── loss_curve.png
│   │   ├── training_plot.png
│   │   ├── confusion_matrix.png
│   │   ├── per_class_metrics.png
│   │   ├── evaluation_results.json
│   │   ├── classification_report.txt
│   │   ├── training_history.json
│   │   └── training_log.csv
│   │
│   ├── templates/
│   └── app.py
│
├── train_model.py
├── evaluate_model.py
├── split_data.py
├── Procfile
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/NicolasAlvarez25/gtzan-ia-corhuila.git
cd gtzan-ia-corhuila
```

### 2. Instalar dependencias

```bash
pip install -r web_app/requirements.txt
```

### 3. Entrenar el modelo

```bash
python train_model.py
```

El script se encarga de:

1. Verificar las carpetas del dataset.
2. Contar las imágenes disponibles.
3. Cargar los conjuntos de entrenamiento y validación.
4. Generar el archivo `clases.json`.
5. Construir la CNN.
6. Ejecutar el entrenamiento.
7. Guardar el modelo con mejor accuracy de validación.
8. Evaluar las 900 imágenes de validación.
9. Generar las métricas y gráficas.
10. Guardar los resultados en JSON, TXT y CSV.

> El entrenamiento puede requerir recursos computacionales elevados si se ejecuta únicamente con CPU.

### 4. Evaluar el modelo guardado

```bash
python evaluate_model.py
```

### 5. Ejecutar la aplicación web

```bash
python web_app/app.py
```

Abrir en el navegador:

```text
http://localhost:5000
```

---

## ☁️ Entrenamiento en Google Colab

Para entrenar el modelo en Google Colab se recomienda seleccionar una GPU T4:

```text
Entorno de ejecución
→ Cambiar tipo de entorno de ejecución
→ Acelerador por hardware
→ GPU T4
```

Después se puede clonar el repositorio:

```python
!git clone https://github.com/NicolasAlvarez25/gtzan-ia-corhuila.git
%cd /content/gtzan-ia-corhuila
```

Comprobar la GPU:

```python
!nvidia-smi
```

Instalar las dependencias:

```python
!pip install -r web_app/requirements.txt
```

Ejecutar el entrenamiento:

```python
!python train_model.py
```

---

## 🌐 Rutas de la aplicación

| Ruta | Funcionalidad |
|---|---|
| `/` | Clasificador de archivos de audio `.wav` |
| `/resultados` | Visualización de métricas y gráficas |
| `/health` | Estado del servicio en formato JSON |

---

## 🔄 Flujo de clasificación

```text
Archivo WAV
    ↓
Procesamiento del fragmento de audio
    ↓
Generación del espectrograma Mel
    ↓
Redimensionamiento a 128 × 128 píxeles
    ↓
Conversión a escala de grises
    ↓
Normalización
    ↓
Predicción de la CNN
    ↓
Probabilidades Softmax
    ↓
Género musical predicho
```

---

## ⚠️ Limitaciones

Aunque el modelo alcanzó una exactitud del 90,56 %, todavía presenta algunas limitaciones:

- Clasifica únicamente nueve géneros.
- Jazz no fue incluido.
- Rock continúa siendo la clase con menor F1-score.
- Las canciones que mezclan varios géneros pueden producir predicciones ambiguas.
- El resultado depende de que los audios externos sean procesados de la misma forma que los datos de entrenamiento.
- El rendimiento sobre canciones externas puede ser diferente al obtenido en el conjunto de validación.
- El dataset GTZAN puede contener similitudes y características propias de sus grabaciones.
- La evaluación actual utiliza fragmentos pertenecientes al dataset procesado.

---

## 🔮 Posibles mejoras futuras

- Incorporar el género Jazz.
- Evaluar con canciones externas al dataset.
- Separar entrenamiento y validación por canción original antes de generar fragmentos.
- Implementar validación cruzada.
- Probar características como MFCC, Chroma y Spectral Contrast.
- Implementar técnicas como SpecAugment.
- Comparar la CNN con EfficientNet, MobileNet o ResNet.
- Clasificar varios fragmentos de una canción y utilizar votación.
- Optimizar el tiempo de inferencia.
- Desplegar la aplicación en un servicio web.
- Ampliar el dataset con nuevas grabaciones.

---

## 📝 Conclusión

La versión optimizada demostró que una CNN puede aprender patrones relevantes en espectrogramas Mel y utilizarlos para clasificar géneros musicales.

La primera versión alcanzaba una exactitud del 75 %. Después de incorporar Batch Normalization, Spatial Dropout, Global Average Pooling, AdamW, Model Checkpoint, Early Stopping y reducción dinámica del learning rate, el modelo alcanzó una exactitud del **90,56 %** y un F1-score macro de **0,9064**.

La mejora obtenida fue de **15,56 puntos porcentuales**, evidenciando el impacto de una arquitectura mejor regularizada y de una estrategia de entrenamiento adaptativa.

---

## 🎓 Información académica

**Proyecto final de Inteligencia Artificial**  
**Corporación Universitaria del Huila, CORHUILA**  
**Periodo académico: 2026-2**

### Integrantes

- Nicolás Gabriel Álvarez Aguirre
- Sergio Luis Angel Romero
