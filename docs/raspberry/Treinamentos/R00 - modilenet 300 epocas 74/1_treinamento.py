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
print("Definindo a camada de Data Augmentation...")
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Construção do Modelo
print("Construindo o modelo com Data Augmentation...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output_layer = Dense(2, activation='softmax')(x) # Garanta 2 classes
model = Model(inputs, outputs=output_layer)

# Compilar o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # O padrão é 0.001
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Treinar o modelo (com mais épocas, pois a tarefa ficou mais difícil para ele)
print("Iniciando o treinamento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300 # Aumentado para 30 épocas
)

# Salvar o modelo treinado
model.save('meu_modelo.keras')

print("\nModelo treinado com sucesso como 'meu_modelo.keras'!")