# CNN Model for Image Classification

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model is trained on a dataset stored in `.npz` files and saved to Google Drive.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Google Colab (for Google Drive integration)

## Installation

1. Install the required libraries:
    ```bash
    pip install tensorflow numpy scikit-learn
    ```

2. Mount Google Drive in Google Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Data Preparation

1. Load the training and testing data from `.npz` files stored in Google Drive:
    ```python
    train_data_path = '/content/drive/MyDrive/train.npz'
    test_data_path = '/content/drive/MyDrive/test.npz'

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    X_train, y_train = train_data['data'], train_data['labels']
    X_test, y_test = test_data['data'], test_data['labels']
    ```

2. Resize the images to the desired input shape and normalize the pixel values:
    ```python
    input_shape = (64, 64, X_train.shape[-1])
    X_train_resized = np.array([tf.image.resize(img, input_shape[:2]).numpy() for img in X_train])
    X_test_resized = np.array([tf.image.resize(img, input_shape[:2]).numpy() for img in X_test])

    X_train_resized = X_train_resized / 255.0
    X_test_resized = X_test_resized / 255.0
    ```

3. Apply data augmentation:
    ```python
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train_resized)
    ```

## Model Architecture

1. Define the CNN model:
    ```python
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    ```

2. Compile the model:
    ```python
    model.compile(optimizer=Adam(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```

## Training

1. Split the training data into training and validation sets:
    ```python
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_resized, y_train, test_size=0.2, random_state=42)
    ```

2. Train the model with early stopping and learning rate reduction:
    ```python
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(datagen.flow(X_train_final, y_train_final, batch_size=32),
                        epochs=30,
                        validation_data=datagen.flow(X_val, y_val, batch_size=32),
                        callbacks=[early_stopping, lr_scheduler])
    ```

## Evaluation

1. Evaluate the model on the test set:
    ```python
    test_loss, test_accuracy = model.evaluate(X_test_resized, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    ```

## Saving and Loading the Model

1. Save the trained model to Google Drive:
    ```python
    model.save('/content/drive/MyDrive/CNN_or_s.h5')
    ```

2. Load the saved model:
    ```python
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    loaded_model = load_model('/content/drive/MyDrive/CNN_or_s.h5')
    ```

3. Evaluate the loaded model:
    ```python
    loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test_resized, y_test)
    print(f"Loaded Model Test Accuracy: {loaded_test_accuracy * 100:.2f}%")
    ```

## Conclusion

This project demonstrates how to build, train, and evaluate a CNN for image classification using TensorFlow and Keras. The model is trained on a dataset stored in `.npz` files and saved to Google Drive for future use.
