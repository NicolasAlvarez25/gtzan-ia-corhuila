"""
==============================================================================
PROYECTO FINAL INTELIGENCIA ARTIFICIAL - CORHUILA (2026-2)
Clasificación de Géneros Musicales con Red Neuronal Convolucional
Módulo de entrenamiento y evaluación del modelo
==============================================================================
"""

import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import callbacks, layers, models


# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_TRAIN = BASE_DIR / "dataset" / "procesado" / "train"
DATASET_VAL = BASE_DIR / "dataset" / "procesado" / "val"

MODEL_DIR = BASE_DIR / "web_app" / "model"
STATIC_DIR = BASE_DIR / "web_app" / "static"

MODEL_SAVE_PATH = MODEL_DIR / "modelo_final.keras"
CLASSES_JSON_PATH = MODEL_DIR / "clases.json"

HISTORY_JSON_PATH = STATIC_DIR / "training_history.json"
EVALUATION_JSON_PATH = STATIC_DIR / "evaluation_results.json"
CLASSIFICATION_REPORT_PATH = STATIC_DIR / "classification_report.txt"
TRAINING_LOG_PATH = STATIC_DIR / "training_log.csv"

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

# EarlyStopping puede detener el entrenamiento antes de llegar a este número.
EPOCHS = 80

INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.000001

SEED = 42

CLASES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "metal",
    "pop",
    "reggae",
    "rock",
]


# ==============================================================================
# 2. REPRODUCIBILIDAD Y GPU
# ==============================================================================

def configurar_entorno():
    """
    Configura semillas aleatorias, crea directorios y comprueba la GPU.
    """

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONFIGURACIÓN DEL ENTORNO")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(f"GPU detectada: {gpus}")

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                print(f"No se pudo activar memory growth: {error}")
    else:
        print("ADVERTENCIA: no se detectó GPU.")
        print("El entrenamiento se realizará en CPU y será más lento.")

    print(f"Dataset de entrenamiento: {DATASET_TRAIN}")
    print(f"Dataset de validación: {DATASET_VAL}")
    print()


# ==============================================================================
# 3. VALIDACIÓN DEL DATASET
# ==============================================================================

def contar_imagenes(ruta):
    """
    Cuenta imágenes compatibles dentro de una carpeta y sus subcarpetas.
    """

    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    return sum(
        1
        for archivo in ruta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    )


def validar_dataset():
    """
    Comprueba que las carpetas de entrenamiento y validación existan.
    También verifica que las nueve clases estén presentes.
    """

    if not DATASET_TRAIN.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de entrenamiento: {DATASET_TRAIN}"
        )

    if not DATASET_VAL.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de validación: {DATASET_VAL}"
        )

    clases_train = sorted(
        carpeta.name
        for carpeta in DATASET_TRAIN.iterdir()
        if carpeta.is_dir()
    )

    clases_val = sorted(
        carpeta.name
        for carpeta in DATASET_VAL.iterdir()
        if carpeta.is_dir()
    )

    if clases_train != CLASES:
        raise ValueError(
            "Las clases de entrenamiento no coinciden.\n"
            f"Esperadas: {CLASES}\n"
            f"Encontradas: {clases_train}"
        )

    if clases_val != CLASES:
        raise ValueError(
            "Las clases de validación no coinciden.\n"
            f"Esperadas: {CLASES}\n"
            f"Encontradas: {clases_val}"
        )

    cantidad_train = contar_imagenes(DATASET_TRAIN)
    cantidad_val = contar_imagenes(DATASET_VAL)

    print("=" * 70)
    print("VALIDACIÓN DEL DATASET")
    print("=" * 70)
    print(f"Imágenes de entrenamiento: {cantidad_train}")
    print(f"Imágenes de validación: {cantidad_val}")
    print(f"Clases detectadas: {len(clases_train)}")
    print()

    if cantidad_train == 0:
        raise ValueError("La carpeta de entrenamiento no contiene imágenes.")

    if cantidad_val == 0:
        raise ValueError("La carpeta de validación no contiene imágenes.")

    return cantidad_train, cantidad_val


# ==============================================================================
# 4. CARGA Y PREPROCESAMIENTO
# ==============================================================================

def cargar_datasets():
    """
    Carga las imágenes desde las carpetas train y val.

    Las imágenes se cargan en escala de grises y se redimensionan a 128 x 128.
    """

    print("=" * 70)
    print("CARGANDO DATASETS")
    print("=" * 70)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TRAIN,
        labels="inferred",
        label_mode="int",
        class_names=CLASES,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_VAL,
        labels="inferred",
        label_mode="int",
        class_names=CLASES,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    if class_names != CLASES:
        raise ValueError(
            f"Orden incorrecto de clases: {class_names}. "
            f"Se esperaba: {CLASES}"
        )

    mapping = {
        str(indice): clase
        for indice, clase in enumerate(class_names)
    }

    with open(CLASSES_JSON_PATH, "w", encoding="utf-8") as archivo:
        json.dump(mapping, archivo, indent=4, ensure_ascii=False)

    print(f"Mapeo de clases guardado en: {CLASSES_JSON_PATH}")

    autotune = tf.data.AUTOTUNE

    # cache() evita volver a leer todas las imágenes del disco en cada época.
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(
        buffer_size=1000,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.prefetch(buffer_size=autotune)

    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(buffer_size=autotune)

    print("Datasets preparados correctamente.")
    print()

    return train_ds, val_ds


# ==============================================================================
# 5. ARQUITECTURA CNN
# ==============================================================================

def bloque_convolucional(modelo, filtros, dropout=0.0):
    """
    Añade un bloque con dos convoluciones, normalización y MaxPooling.
    """

    modelo.add(
        layers.Conv2D(
            filtros,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        )
    )
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Activation("relu"))

    modelo.add(
        layers.Conv2D(
            filtros,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        )
    )
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Activation("relu"))

    modelo.add(layers.MaxPooling2D(pool_size=(2, 2)))

    if dropout > 0:
        modelo.add(layers.SpatialDropout2D(dropout))


def construir_cnn():
    """
    Construye una CNN optimizada para espectrogramas Mel de 128 x 128.
    """

    print("=" * 70)
    print("CONSTRUYENDO MODELO CNN")
    print("=" * 70)

    # En espectrogramas no se utilizan volteos horizontales o verticales.
    # El desplazamiento horizontal representa pequeños cambios en el tiempo.
    data_augmentation = models.Sequential(
        [
            layers.RandomTranslation(
                height_factor=0.0,
                width_factor=0.08,
                fill_mode="nearest",
                seed=SEED,
            ),
            layers.RandomContrast(
                factor=0.10,
                seed=SEED,
            ),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(name="cnn_generos_musicales")

    model.add(
        layers.Input(
            shape=(IMG_SIZE[0], IMG_SIZE[1], 1),
            name="espectrograma_entrada",
        )
    )

    model.add(data_augmentation)
    model.add(layers.Rescaling(1.0 / 255.0, name="normalizacion"))

    bloque_convolucional(model, filtros=32, dropout=0.10)
    bloque_convolucional(model, filtros=64, dropout=0.15)
    bloque_convolucional(model, filtros=128, dropout=0.25)
    bloque_convolucional(model, filtros=256, dropout=0.30)

    # Reduce parámetros en comparación con Flatten y disminuye el sobreajuste.
    model.add(layers.GlobalAveragePooling2D(name="promedio_global"))

    model.add(
        layers.Dense(
            256,
            use_bias=False,
            name="capa_densa",
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.45))

    model.add(
        layers.Dense(
            len(CLASES),
            activation="softmax",
            dtype="float32",
            name="clasificacion",
        )
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=INITIAL_LEARNING_RATE,
        weight_decay=0.0001,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    print("Modelo construido correctamente.")
    print()

    return model


# ==============================================================================
# 6. CALLBACKS
# ==============================================================================

def crear_callbacks():
    """
    Configura guardado del modelo, parada temprana y reducción del learning rate.
    """

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    cb_early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    )

    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=4,
        min_delta=0.001,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    )

    cb_csv_logger = callbacks.CSVLogger(
        filename=TRAINING_LOG_PATH,
        separator=",",
        append=False,
    )

    cb_terminate_nan = callbacks.TerminateOnNaN()

    return [
        cb_checkpoint,
        cb_early_stopping,
        cb_reduce_lr,
        cb_csv_logger,
        cb_terminate_nan,
    ]


# ==============================================================================
# 7. GRÁFICAS DEL ENTRENAMIENTO
# ==============================================================================

def guardar_historial(history):
    """
    Guarda el historial numérico y genera las curvas de accuracy y loss.
    """

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    learning_rates = history.history.get("learning_rate", [])

    epocas_ejecutadas = len(acc)
    epochs_range = range(1, epocas_ejecutadas + 1)

    historial = {
        "epochs_configured": EPOCHS,
        "epochs_executed": epocas_ejecutadas,
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "image_size": list(IMG_SIZE),
        "accuracy": [float(valor) for valor in acc],
        "val_accuracy": [float(valor) for valor in val_acc],
        "loss": [float(valor) for valor in loss],
        "val_loss": [float(valor) for valor in val_loss],
        "learning_rate": [
            float(valor)
            for valor in learning_rates
        ],
    }

    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, indent=4, ensure_ascii=False)

    print(f"Historial guardado en: {HISTORY_JSON_PATH}")

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs_range,
        acc,
        color="royalblue",
        linewidth=2,
        label="Entrenamiento",
    )
    plt.plot(
        epochs_range,
        val_acc,
        color="crimson",
        linewidth=2,
        label="Validación",
    )
    plt.title("Evolución de la precisión")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        STATIC_DIR / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs_range,
        loss,
        color="royalblue",
        linewidth=2,
        label="Entrenamiento",
    )
    plt.plot(
        epochs_range,
        val_loss,
        color="crimson",
        linewidth=2,
        label="Validación",
    )
    plt.title("Evolución de la pérdida")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        STATIC_DIR / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        epochs_range,
        acc,
        color="royalblue",
        linewidth=2,
        label="Entrenamiento",
    )
    axes[0].plot(
        epochs_range,
        val_acc,
        color="crimson",
        linewidth=2,
        label="Validación",
    )
    axes[0].set_title("Precisión")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(
        epochs_range,
        loss,
        color="royalblue",
        linewidth=2,
        label="Entrenamiento",
    )
    axes[1].plot(
        epochs_range,
        val_loss,
        color="crimson",
        linewidth=2,
        label="Validación",
    )
    axes[1].set_title("Pérdida")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        STATIC_DIR / "training_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Curvas de entrenamiento generadas correctamente.")


# ==============================================================================
# 8. EVALUACIÓN DEL MODELO
# ==============================================================================

def obtener_predicciones(model, val_ds):
    """
    Ejecuta la predicción del conjunto completo de validación.
    """

    y_true = []
    y_pred = []
    y_probabilities = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted_labels.tolist())
        y_probabilities.extend(predictions.tolist())

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_probabilities),
    )


def generar_matriz_confusion(y_true, y_pred):
    """
    Genera y guarda la matriz de confusión.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASES))),
    )

    plt.figure(figsize=(11, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASES,
        yticklabels=CLASES,
        cbar=True,
    )

    plt.title("Matriz de confusión del modelo")
    plt.xlabel("Género predicho")
    plt.ylabel("Género real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    matriz_path = STATIC_DIR / "confusion_matrix.png"

    plt.savefig(
        matriz_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Matriz de confusión guardada en: {matriz_path}")

    return cm


def generar_grafica_por_clase(reporte_dict):
    """
    Genera una gráfica de precision, recall y F1-score por género.
    """

    precision_por_clase = [
        reporte_dict[clase]["precision"]
        for clase in CLASES
    ]

    recall_por_clase = [
        reporte_dict[clase]["recall"]
        for clase in CLASES
    ]

    f1_por_clase = [
        reporte_dict[clase]["f1-score"]
        for clase in CLASES
    ]

    posiciones = np.arange(len(CLASES))
    ancho = 0.25

    plt.figure(figsize=(14, 6))

    plt.bar(
        posiciones - ancho,
        precision_por_clase,
        width=ancho,
        label="Precision",
    )
    plt.bar(
        posiciones,
        recall_por_clase,
        width=ancho,
        label="Recall",
    )
    plt.bar(
        posiciones + ancho,
        f1_por_clase,
        width=ancho,
        label="F1-score",
    )

    plt.title("Métricas por género musical")
    plt.xlabel("Género")
    plt.ylabel("Resultado")
    plt.xticks(posiciones, CLASES, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    metricas_path = STATIC_DIR / "per_class_metrics.png"

    plt.savefig(
        metricas_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Métricas por clase guardadas en: {metricas_path}")


def evaluar_modelo(model, val_ds, history):
    """
    Evalúa el mejor modelo y guarda métricas, reportes y gráficas.
    """

    print()
    print("=" * 70)
    print("EVALUACIÓN DEL MEJOR MODELO")
    print("=" * 70)

    loss, keras_accuracy = model.evaluate(val_ds, verbose=1)

    y_true, y_pred, _ = obtener_predicciones(model, val_ds)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    reporte_texto = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASES))),
        target_names=CLASES,
        digits=4,
        zero_division=0,
    )

    reporte_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASES))),
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as archivo:
        archivo.write(reporte_texto)

    cm = generar_matriz_confusion(y_true, y_pred)
    generar_grafica_por_clase(reporte_dict)

    mejor_val_accuracy = max(
        history.history.get("val_accuracy", [accuracy])
    )

    mejor_epoca = (
        int(
            np.argmax(
                history.history.get("val_accuracy", [accuracy])
            )
        )
        + 1
    )

    resultados = {
        "model": MODEL_SAVE_PATH.name,
        "classes": CLASES,
        "validation_samples": int(len(y_true)),
        "loss": float(loss),
        "keras_accuracy": float(keras_accuracy),
        "accuracy": float(accuracy),
        "accuracy_percentage": float(accuracy * 100),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "best_validation_accuracy": float(mejor_val_accuracy),
        "best_epoch": mejor_epoca,
        "epochs_configured": EPOCHS,
        "epochs_executed": len(
            history.history.get("accuracy", [])
        ),
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "image_size": list(IMG_SIZE),
        "confusion_matrix": cm.tolist(),
        "classification_report": reporte_dict,
    }

    with open(
        EVALUATION_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resultados,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(reporte_texto)
    print("-" * 70)
    print(f"Loss:             {loss:.4f}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"Precision macro:  {precision_macro:.4f}")
    print(f"Recall macro:     {recall_macro:.4f}")
    print(f"F1-score macro:   {f1_macro:.4f}")
    print(f"Mejor época:      {mejor_epoca}")
    print("-" * 70)

    print(
        f"Reporte de clasificación guardado en: "
        f"{CLASSIFICATION_REPORT_PATH}"
    )

    print(
        f"Resultados JSON guardados en: "
        f"{EVALUATION_JSON_PATH}"
    )

    return resultados


# ==============================================================================
# 9. EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    configurar_entorno()
    cantidad_train, cantidad_val = validar_dataset()

    train_ds, val_ds = cargar_datasets()

    model = construir_cnn()
    model.summary()

    lista_callbacks = crear_callbacks()

    print()
    print("=" * 70)
    print("INICIANDO ENTRENAMIENTO DE LA RED NEURONAL")
    print("=" * 70)
    print(f"Imágenes de entrenamiento: {cantidad_train}")
    print(f"Imágenes de validación: {cantidad_val}")
    print(f"Épocas máximas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate inicial: {INITIAL_LEARNING_RATE}")
    print()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=lista_callbacks,
        verbose=1,
    )

    print()
    print("=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    guardar_historial(history)

    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo guardado por ModelCheckpoint."
        )

    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    resultados = evaluar_modelo(
        best_model,
        val_ds,
        history,
    )

    print()
    print("=" * 70)
    print("PROCESO FINALIZADO CORRECTAMENTE")
    print("=" * 70)
    print(f"Modelo: {MODEL_SAVE_PATH}")
    print(f"Clases: {CLASSES_JSON_PATH}")
    print(f"Resultados: {EVALUATION_JSON_PATH}")
    print(f"Gráficas: {STATIC_DIR}")
    print(
        f"Accuracy final: "
        f"{resultados['accuracy_percentage']:.2f}%"
    )


if __name__ == "__main__":
    main()
