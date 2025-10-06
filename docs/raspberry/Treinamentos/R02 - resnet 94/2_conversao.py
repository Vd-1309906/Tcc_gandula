import tensorflow as tf
import numpy as np
# --- PASSO 1: Importar a função que o Keras não está encontrando ---
from tensorflow.keras.applications.resnet_v2 import preprocess_input

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
  # Note que a função preprocess_input já está dentro do modelo,
  # então não precisamos aplicá-la aqui.
  for images, _ in train_ds.take(100):
    for image in images:
      yield [np.expand_dims(image, axis=0).astype(np.float32)]

# --- Carregar o Modelo Keras Salvo (COM A CORREÇÃO) ---
print("Carregando o modelo treinado .keras...")
MODEL_FILENAME = 'meu_modelo_resnet.keras' # <<< Corrigido para o nome do arquivo do ResNet

# --- PASSO 2: Usar o argumento 'custom_objects' ---
model = tf.keras.models.load_model(
    MODEL_FILENAME,
    custom_objects={'preprocess_input': preprocess_input}
)
print("Modelo carregado com sucesso!")

# --- Converter o Modelo para TFLite com Quantização ---
print("Iniciando a conversão para TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# O input do modelo Keras é float32, mas o modelo TFLite final terá entrada uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# --- Salvar o Modelo TFLite ---
TFLITE_FILENAME = 'meu_modelo_resnet_quantizado.tflite' # <<< Nome do arquivo final atualizado
with open(TFLITE_FILENAME, 'wb') as f:
  f.write(tflite_model_quant)

print(f"\nModelo convertido com sucesso e salvo como '{TFLITE_FILENAME}'!")