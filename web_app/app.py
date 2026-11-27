import os
import json
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request
import tensorflow as tf
import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
STATIC_SPECS = BASE_DIR / 'static' / 'specs'
MODEL_PATH = BASE_DIR / 'model' / 'modelo_final.keras'
CLASSES_PATH = BASE_DIR / 'model' / 'clases.json'
EVALUATION_PATH = BASE_DIR / 'static' / 'evaluation_results.json'
HISTORY_PATH = BASE_DIR / 'static' / 'training_history.json'

UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_SPECS.mkdir(parents=True, exist_ok=True)

# Cargar modelo y clases
print("Cargando modelo...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Modelo cargado exitosamente")


def detectar_rescaling(modelo):
    for layer in modelo.layers:
        if 'rescaling' in layer.__class__.__name__.lower():
            return True
        if hasattr(layer, 'layers') and detectar_rescaling(layer):
            return True
    return False


TIENE_RESCALING = detectar_rescaling(model)
print(f"Capa Rescaling interna detectada: {TIENE_RESCALING}")

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    clases_dict = json.load(f)

CLASES = [clases_dict[str(i)] for i in range(len(clases_dict))]
print(f"Clases cargadas: {CLASES}")

IMG_SIZE = (128, 128)
DURACION_AUDIO = 3.0


def cargar_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def normalizar_evaluacion(raw, clases_fallback, umbral=0.80):
    data = dict(raw) if isinstance(raw, dict) else {}
    clases = data.get('classes') or data.get('clases') or clases_fallback
    reporte = data.get('classification_report', {})

    metricas_por_clase = {}
    existente = data.get('metricas_por_clase')
    if isinstance(existente, dict) and existente:
        metricas_por_clase = existente
    else:
        for clase in clases:
            m = reporte.get(clase, {})
            f1 = m.get('f1-score', m.get('f1_score', m.get('f1', 0.0)))
            soporte = int(m.get('support', m.get('soporte', 0)))
            metricas_por_clase[clase] = {
                'precision': float(m.get('precision', 0.0)),
                'recall': float(m.get('recall', 0.0)),
                'f1_score': float(f1),
                'f1': float(f1),
                'f1-score': float(f1),
                'support': soporte,
                'soporte': soporte,
            }

    macro = reporte.get('macro avg', {})
    accuracy = float(data.get('accuracy', data.get('keras_accuracy', data.get('exactitud', 0.0))))
    acc_pct = round(float(data.get('accuracy_percentage', data.get('exactitud_porcentaje', accuracy * 100))), 2)
    loss_val = round(float(data.get('loss', data.get('perdida', 0.0))), 4)
    epocas_val = int(data.get('epochs_executed', data.get('epochs_configured', 0)))
    epocas_conf = int(data.get('epochs_configured', epocas_val))
    lr_val = float(data.get('initial_learning_rate', data.get('learning_rate', 0.0)))
    muestras_val = int(data.get('validation_samples', data.get('muestras_validacion', 900)))

    precision_macro = float(data.get('precision_macro', data.get('precision', macro.get('precision', 0.0))))
    recall_macro = float(data.get('recall_macro', data.get('recall', macro.get('recall', 0.0))))
    f1_macro = float(data.get('f1_macro', data.get('f1_score', macro.get('f1-score', 0.0))))

    data['metricas_por_clase'] = metricas_por_clase
    data.setdefault('umbral_requerido', umbral)
    data['cumple_umbral'] = accuracy >= float(data['umbral_requerido'])

    for k in ('accuracy_validacion', 'accuracy_porcentaje', 'exactitud_validacion', 'exactitud_porcentaje'):
        data.setdefault(k, acc_pct)
    data.setdefault('val_accuracy', round(accuracy, 4))
    data.setdefault('exactitud', round(accuracy, 4))

    for k in ('loss_validacion', 'val_loss', 'perdida_validacion', 'perdida'):
        data.setdefault(k, loss_val)

    for k in ('epocas', 'epocas_entrenamiento', 'epocas_ejecutadas', 'epochs'):
        data.setdefault(k, epocas_val)
    data.setdefault('epocas_configuradas', epocas_conf)
    data.setdefault('mejor_epoca', int(data.get('best_epoch', 0)))

    for k in ('learning_rate', 'lr', 'tasa_aprendizaje'):
        data.setdefault(k, lr_val)

    for k in ('validation_samples', 'imagenes_validacion', 'total_validacion', 'muestras_validacion'):
        data.setdefault(k, muestras_val)

    data.setdefault('modelo', data.get('model', MODEL_PATH.name))
    data.setdefault('clases', clases)
    data.setdefault('precision', precision_macro)
    data.setdefault('recall', recall_macro)
    data.setdefault('f1_score', f1_macro)

    return data


def normalizar_historial(raw, evaluacion=None):
    if not isinstance(raw, dict):
        return {}
    data = dict(raw.get('history', raw))

    acc = data.get('accuracy', data.get('acc', []))
    val_acc = data.get('val_accuracy', data.get('val_acc', []))
    loss = data.get('loss', [])
    val_loss = data.get('val_loss', [])

    data.setdefault('accuracy', acc)
    data.setdefault('val_accuracy', val_acc)
    data.setdefault('loss', loss)
    data.setdefault('val_loss', val_loss)
    data.setdefault('epocas', data.get('epochs', list(range(1, len(acc) + 1))))
    return data


def segmento_a_tensor(y_seg, sr, guardar_imagen=False):
    """Genera el espectrograma 432x288 con plt.imshow idéntico al dataset de entrenamiento"""
    S = librosa.feature.melspectrogram(
        y=y_seg,
        sr=sr,
        n_mels=128
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(6, 4), dpi=72)
    plt.imshow(S_dB)

    spec_filename = f'spectrogram_{uuid.uuid4().hex}.jpg'
    spec_path = STATIC_SPECS / spec_filename

    plt.savefig(
        spec_path,
        format='jpg',
        dpi=72
    )
    plt.close(fig)

    img = Image.open(spec_path).convert('L')
    img_resized = img.resize(IMG_SIZE)

    img_array = np.array(img_resized, dtype=np.float32)
    if not TIENE_RESCALING:
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    if not guardar_imagen:
        try:
            os.remove(spec_path)
        except OSError:
            pass
        return img_array, None

    return img_array, f'specs/{spec_filename}'


def audio_to_spectrograms(audio_path):
    """Extrae hasta 5 fragmentos de 3 segundos distribuidos a lo largo del audio"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    samples_por_ventana = int(DURACION_AUDIO * sr)

    if len(y) <= samples_por_ventana:
        y_pad = np.pad(y, (0, max(0, samples_por_ventana - len(y))))
        tensor, spec_url = segmento_a_tensor(y_pad, sr, guardar_imagen=True)
        return [tensor], spec_url

    inicio = samples_por_ventana if len(y) > samples_por_ventana * 4 else 0
    fin = len(y) - samples_por_ventana

    num_ventanas = min(5, max(1, (fin - inicio) // samples_por_ventana + 1))
    posiciones = np.linspace(inicio, fin, num=num_ventanas, dtype=int)
    indice_central = len(posiciones) // 2

    tensores = []
    spec_url_central = None

    for idx, pos in enumerate(posiciones):
        segmento = y[pos:pos + samples_por_ventana]
        es_central = (idx == indice_central)
        tensor, spec_url = segmento_a_tensor(segmento, sr, guardar_imagen=es_central)
        tensores.append(tensor)
        if es_central:
            spec_url_central = spec_url

    return tensores, spec_url_central


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resultados')
def resultados():
    evaluacion = normalizar_evaluacion(cargar_json(EVALUATION_PATH), CLASES)
    historial = normalizar_historial(cargar_json(HISTORY_PATH), evaluacion)

    reporte_txt = ''
    reporte_path = BASE_DIR / 'static' / 'classification_report.txt'
    if reporte_path.exists():
        reporte_txt = reporte_path.read_text(encoding='utf-8')

    return render_template(
        'resultados.html',
        evaluacion=evaluacion,
        historial=historial,
        reporte_txt=reporte_txt
    )


@app.route('/health')
def health():
    return {
        'status': 'ok',
        'modelo': MODEL_PATH.name,
        'clases': CLASES,
        'rescaling_interno': TIENE_RESCALING
    }


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return render_template('index.html', error='No se seleccionó ningún archivo')

        file = request.files['audio']

        if file.filename == '':
            return render_template('index.html', error='No se seleccionó ningún archivo')

        if not file.filename.lower().endswith('.wav'):
            return render_template('index.html', error='Solo se permiten archivos .WAV')

        filename = f'{uuid.uuid4().hex}.wav'
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        tensores, spec_url = audio_to_spectrograms(str(filepath))

        lote = np.vstack(tensores)
        predicciones_ventanas = model.predict(lote, verbose=0)
        predictions = np.mean(predicciones_ventanas, axis=0)

        top_indices = predictions.argsort()[-3:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'genre': CLASES[idx],
                'confidence': round(float(predictions[idx]) * 100, 2)
            })

        try:
            os.remove(filepath)
        except OSError:
            pass

        return render_template(
            'index.html',
            results=results,
            spec_image=spec_url
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f'Error al procesar el audio: {str(e)}'
        )


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
