import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks

class PalmVeinTrainer:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=16):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        image = cv2.resize(image, self.img_size)

        # Enhance vein pattern
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)

        # Normalize
        normalized = enhanced / 255.0
        return np.stack([normalized]*3, axis=-1)  # Convert grayscale to 3-channel RGB

    def load_dataset(self):
        X, y, class_names = [], [], []
        print("Loading and preprocessing dataset...")
        for idx, person in enumerate(sorted(os.listdir(self.dataset_path))):
            person_dir = os.path.join(self.dataset_path, person)
            if os.path.isdir(person_dir):
                class_names.append(person)
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(idx)

        X = np.array(X)
        y = np.array(y)
        os.makedirs('models', exist_ok=True)
        with open('models/class_names.txt', 'w') as f:
            for name in class_names:
                f.write(name + '\n')
        return X, y, len(class_names)

    def build_model(self, num_classes):
        base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                  include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze base

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, epochs=50):
        X, y, num_classes = self.load_dataset()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_gen = datagen.flow(X_train, y_train, batch_size=self.batch_size)

        model = self.build_model(num_classes)

        cb = [
            callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
        ]

        history = model.fit(train_gen, epochs=epochs, validation_data=(X_val, y_val),
                            callbacks=cb, verbose=1)

        model.save('models/final_model.h5')
        self.plot_history(history)
        return model

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('models/training_plot.png')
        plt.close()

if __name__ == "__main__":
    trainer = PalmVeinTrainer(dataset_path='dataset')
    trainer.train()