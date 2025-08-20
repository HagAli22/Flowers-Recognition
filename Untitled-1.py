# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# %%
import kagglehub
import os
# Download latest version
path = kagglehub.dataset_download("alxmamaev/flowers-recognition")

print("Path to dataset files:", path)

# %%
def creat_my_data(dataset_path):
  X = []
  y = []
  for cls in os.listdir(dataset_path):
    images_paths = os.listdir(f'{dataset_path}/{cls}')
    images_paths = [f'{dataset_path}/{cls}/{img}' for img in images_paths]
    labels = [cls] * len(images_paths)
    X.extend(images_paths)
    y.extend(labels)
  return X, y

# %%
x,y = creat_my_data("/kaggle/input/flowers-recognition/flowers")

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True, random_state=42)

# %%
train_df = pd.DataFrame({'image_path': x_train, 'label': y_train})
test_df = pd.DataFrame({'image_path': x_test, 'label': y_test})

# %%
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
val_datagen=ImageDataGenerator(
    rescale=1./255,
)

# %%
train = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Load validation images
val = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# %% [markdown]
# # Model

# %%
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(0.01), input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully connected
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer for 5 classes
model.add(Dense(5, activation='softmax'))

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# %%
history=model.fit(train, epochs=25, validation_data=val,callbacks=[early_stopping])

# %%
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")

# %%
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()

# %%
model.save('flowers.keras')

# %%



