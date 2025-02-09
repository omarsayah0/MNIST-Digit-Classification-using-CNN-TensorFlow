import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return(x_train, y_train, x_test, y_test)

def set_model():
    model = keras.Sequential([
        layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(10, activation = 'softmax')
    ])
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return (model)

def show_eval(history):
    _, axes = plt.subplots(1, 2, figsize=(12, 5), num="Roc History")
    axes[0].plot(history.history['accuracy'], label='train_acc')
    axes[0].plot(history.history['val_accuracy'], label='val_acc')
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='train_loss')
    axes[1].plot(history.history['val_loss'], label='val_loss')
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.show()

def show_samples(x_test, y_test, y_pred):
    _, axes = plt.subplots(3, 4, figsize = (10, 5), num="Samples")
    axes = axes.flatten()
    for i in range (12):
        axes[i].imshow(x_test[i].reshape(28, 28), cmap = 'gray')
        axes[i].set_title(f"pred: {y_pred[i]}, True: {y_test[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.title("Samples")
    plt.show()

def show_evaluate(y_test, y_pred, y_pred_proba):
    plt.figure(num="Classification Report",figsize=(10, 5))
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report)
    sns.heatmap(report.iloc[: -1, : -2], annot=True, fmt=".2f", cmap="Blues")
    plt.title("Classification report")
    plt.show()
    _, ax = plt.subplots(figsize=(8, 6), num="Confusion Matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    plt.show()
    plt.figure(num="Roc Curve",figsize=(12, 8))
    y_roc = label_binarize(y_test, classes=[i for i in range(10)])
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_roc[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc_score(y_roc[:, i], y_pred_proba[:, i]):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve")
    plt.legend()
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_data()
    model = set_model()
    history = model.fit(x_train, y_train, epochs=5, batch_size = 64, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    show_eval(history)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    show_samples(x_test, y_test, y_pred)
    y_pred_proba = model.predict(x_test)
    show_evaluate(y_test, y_pred, y_pred_proba)
    model.save("cnn.keras")

if __name__ == "__main__":
    main()

