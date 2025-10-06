import tensorflow as tf
import numpy as np
# --- MUDANÇA 1: Remover a importação desnecessária ---
# from tensorflow.keras.applications.resnet_v2 import preprocess_input

# --- Carregar os dados de treino para usar no "representative_dataset" ---
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'dataset/' 

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# --- Função Geradora para o Representative Dataset ---
def representative_data_gen():
  # O modelo MobileNetV2 já tem a camada de Rescaling,
  # então alimentamos com os dados originais.
  for images, _ in train_ds.take(100):
    for image in images:
      yield [np.expand_dims(image, axis=0).astype(np.float32)]

# --- Carregar o Modelo Keras Salvo (COM A CORREÇÃO) ---
print("Carregando o modelo treinado .keras...")
# --- MUDANÇA 2: Usar o nome do arquivo do MobileNet ---
MODEL_FILENAME = 'meu_modelo_mobilenet.keras' 

# --- MUDANÇA 3: Remover o argumento 'custom_objects' ---
model = tf.keras.models.load_model(MODEL_FILENAME)
print("Modelo carregado com sucesso!")

# --- Converter o Modelo para TFLite com Quantização ---
print("Iniciando a conversão para TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# --- Salvar o Modelo TFLite ---
# --- MUDANÇA 4: Salvar com um nome que corresponda ao modelo ---
TFLITE_FILENAME = 'meu_modelo_mobilenet_quantizado.tflite'
with open(TFLITE_FILENAME, 'wb') as f:
  f.write(tflite_model_quant)

print(f"\nModelo convertido com sucesso e salvo como '{TFLITE_FILENAME}'!")
