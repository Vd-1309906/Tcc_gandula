import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # <<< Importar os Callbacks


# Parâmetros
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'dataset/'
TEST_DIR = 'dataset_teste/'

# Carregar dados de TREINO e VALIDAÇÃO
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Otimizar performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Dados prontos!")

# Camada de Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])
# --- Construção do Modelo (Fase Inicial) ---
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False # MANTÉM CONGELADO INICIALMENTE

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output_layer = Dense(2, activation='softmax')(x)
model = Model(inputs, outputs=output_layer)

# Compilar o modelo para a primeira fase
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Taxa de aprendizado inicial
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- Definir os Callbacks ---
# Para o treinamento inicial, podemos usar apenas a parada antecipada
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- TREINAMENTO FASE 1: Treinar apenas a "cabeça" ---
print("Iniciando Treinamento - Fase 1 (Cabeça de Classificação)")
initial_epochs = 20 # Aumente as épocas, o EarlyStopping vai parar se não melhorar
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stopping] # Adiciona o callback
)

# --- FINE-TUNING FASE 2: Descongelar parte do modelo e treinar de novo ---
print("\nIniciando Treinamento - Fase 2 (Fine-Tuning)")
base_model.trainable = True # Descongela o modelo base

# Vamos congelar as primeiras camadas e treinar apenas as últimas
# Um bom número para começar é descongelar as últimas 20 ou 30 camadas
fine_tune_at = len(base_model.layers) - 30 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompilar o modelo com uma TAXA DE APRENDIZADO MUITO BAIXA. Isso é CRUCIAL.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # 100x menor
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Definir callbacks para o Fine-Tuning
# O ReduceLROnPlateau é especialmente útil aqui
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
# Podemos usar o EarlyStopping de novo
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Continua de onde parou
    callbacks=[early_stopping, reduce_lr] # Usando os dois callbacks
)

# Salvar o modelo final, após o fine-tuning
model.save('meu_modelo.keras')
print("\nModelo treinado com Fine-Tuning e salvo com sucesso!")