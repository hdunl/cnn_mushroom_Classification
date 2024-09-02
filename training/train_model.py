import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# enable memory growth to avoid gpu memory issues
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# multi-gpu strategy, used 4x 80Gb A100s in my training
strategy = tf.distribute.MirroredStrategy()
print(f"number of devices: {strategy.num_replicas_in_sync}")

# parameters
IMG_SIZE = (260, 260)
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
EPOCHS = 15
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(directory, is_training=True):
    img_paths, labels = [], []
    class_names = sorted(os.listdir(directory))
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_idx)
    
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())  # ignore errors
    if is_training:
        ds = ds.shuffle(buffer_size=len(img_paths))
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return ds

# load datasets
train_ds = create_dataset('/home/mushroom_project/data/all_species', is_training=True) #REPLACE WITH YOUR PATH
val_ds = create_dataset('/home/mushroom_project/data/all_species', is_training=False) #REPLACE WITH YOUR PATH

# number of classes
num_classes = len(os.listdir('/home/mushroom_project/data/all_species'))

# model creation
with strategy.scope():
    base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001) # reduce learning rate when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # stop training early if validation loss doesn't improve for 5 epochs

# training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping]
)

# save the model
model.save('/home/mushroom_project/mushroom_species_classifier.h5')
print("model saved as 'mushroom_species_classifier.h5'")

# plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('model accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title('model loss')

plt.tight_layout()
plt.savefig('/home/mushroom_project/training_history.png')
print("training history plot saved as 'training_history.png'")
