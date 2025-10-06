import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, Lambda
from tensorflow.keras.models import Model
# --- MUDANÇA 1: Importar o novo modelo e sua função de pré-processamento ---
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
# --- FIM DA MUDANÇA 1 ---
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parâmetros
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'dataset/'

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

# --- Construção do Modelo com ResNet50V2 ---
print("Construindo o modelo com ResNet50V2...")

# A entrada não muda
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# --- MUDANÇA 2: Usar a função de pré-processamento do ResNet ---
# Em vez de Rescaling(1./255), usamos uma camada Lambda para aplicar a função específica do ResNet
x = Lambda(preprocess_input)(inputs)
# --- FIM DA MUDANÇA 2 ---

x = data_augmentation(x)

# --- MUDANÇA 3: Usar o ResNet50V2 como modelo base ---
# Voltamos a usar 'input_shape' que é o padrão para este modelo
base_model = ResNet50V2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# --- FIM DA MUDANÇA 3 ---

base_model.trainable = False

# Conectar a saída do modelo base
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output_layer = Dense(2, activation='softmax')(x) # Garanta que está 2, pois seus dados têm 2 classes
model = Model(inputs, outputs=output_layer)


# --- O resto do script continua exatamente o mesmo ---
# Compilar o modelo para a primeira fase
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# TREINAMENTO FASE 1
print("Iniciando Treinamento - Fase 1 (Cabeça de Classificação)")
initial_epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stopping]
)

# FINE-TUNING FASE 2
print("\nIniciando Treinamento - Fase 2 (Fine-Tuning)")
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 30 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, reduce_lr]
)

# --- MUDANÇA 4: Salvar com um nome diferente ---
model.save('meu_modelo_resnet.keras')
print("\nModelo treinado com ResNet50V2 e salvo com sucesso!")