import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import shutil
# --- MUDANÇA 1: Importar a função que o Keras não está encontrando ---
from tensorflow.keras.applications.resnet_v2 import preprocess_input

print("Iniciando a Ação 1: Análise de Erros")

# --- 1. Preparar Pastas para Salvar os Erros ---
ERROS_DIR = "analise_de_erros"
if os.path.exists(ERROS_DIR):
    shutil.rmtree(ERROS_DIR)
os.makedirs(ERROS_DIR)
print(f"Pasta '{ERROS_DIR}' criada para salvar as imagens classificadas incorretamente.")


# --- 2. Carregar o Conjunto de Teste ---
IMG_SIZE = 224
BATCH_SIZE = 32
TEST_DIR = 'dataset_teste/'

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
image_paths = test_ds.file_paths
print(f"Encontradas {len(image_paths)} imagens de teste em {len(class_names)} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
print("Dados de teste carregados e otimizados!")


# --- 3. Carregar o Modelo e Fazer Previsões ---
# --- MUDANÇA 2: Usar o nome do arquivo correto do modelo ResNet ---
MODEL_PATH = "meu_modelo_resnet.keras" 

# --- MUDANÇA 3: Usar 'custom_objects' para carregar o modelo ---
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'preprocess_input': preprocess_input}
)
print("Modelo carregado com sucesso!")

print("Iniciando avaliação no conjunto de teste...")

predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = np.concatenate([y for x, y in test_ds], axis=0)


# --- 4. Análise de Erros ---
print("Analisando erros e copiando imagens...")
erros_count = 0
for i in range(len(image_paths)):
    caminho_original = image_paths[i]
    rotulo_verdadeiro_idx = y_true[i]
    rotulo_previsto_idx = y_pred[i]

    if rotulo_previsto_idx != rotulo_verdadeiro_idx:
        erros_count += 1
        nome_rotulo_verdadeiro = class_names[rotulo_verdadeiro_idx]
        nome_rotulo_previsto = class_names[rotulo_previsto_idx]
        
        pasta_erro_especifico = os.path.join(ERROS_DIR, f"real_{nome_rotulo_verdadeiro}__previsto_{nome_rotulo_previsto}")
        os.makedirs(pasta_erro_especifico, exist_ok=True)
        
        nome_arquivo = os.path.basename(caminho_original)
        shutil.copy(caminho_original, os.path.join(pasta_erro_especifico, nome_arquivo))

print(f"{erros_count} erros encontrados e salvos em subpastas dentro de '{ERROS_DIR}'.")


# --- 5. Calcular e Exibir as Métricas Finais ---
accuracy = accuracy_score(y_true, y_pred)
print(f"\n========================================================")
print(f"Acurácia Final do Modelo: {accuracy*100:.2f}%")
print(f"========================================================")

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_true, y_pred, target_names=class_names))