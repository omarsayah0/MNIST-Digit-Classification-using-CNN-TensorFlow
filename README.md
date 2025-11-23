# CNN Handwritten Digit Recognition (MNIST)

## About
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset. It demonstrates how to build, train, and evaluate a deep learning CNN for image classification.
The model learns directly from 28×28 grayscale images of handwritten digits and achieves high accuracy through convolution and pooling layers.
It visualizes training performance, evaluates model accuracy, and displays key metrics such as confusion matrix, classification report, and ROC curves.

---

## Files
- `cnn.py` → Python code implementing the model.
- `cnn.keras` → Saved trained model (after running the code).

---

## Steps Included

### 1️⃣ Data Preprocessing
- Load MNIST dataset from keras.datasets.mnist

        Loads 60,000 training and 10,000 testing grayscale images of handwritten digits (0–9).

        This is a standard benchmark dataset for image classification tasks.

- Reshape data to (28, 28, 1) for CNN input

        CNN layers in Keras expect input with a channel dimension (height, width, channels).

        Since MNIST images are grayscale (one channel), we reshape from (28, 28) → (28, 28, 1), This allows the Conv2D layer to correctly interpret the image as 2D spatial data rather than a flat array.

- Convert to float32 because original pixel values are stored as integers (0–255).
  
        Neural networks work best with floating-point precision, improving numerical stability during gradient updates. It ensures consistent computation on GPU/TPU and avoids integer overflow or precision issues.

- Normalize pixel values to range [0, 1]

        Each pixel value is divided by 255.0. Normalization helps the model train faster and more stably by keeping all input values within a small range, preventing large gradients and improving convergence.

---

###  2️⃣ Model Architecture (Simple Explanation)

| **Layer** | **Description** |
|------------|----------------|
| **Conv2D (32 filters, 3×3, ReLU)** | Finds basic patterns like edges and lines in the image. |
| **MaxPooling2D (2×2)** | Makes the image smaller and keeps only the most important features. |
| **Conv2D (64 filters, 3×3, ReLU)** | Learns more complex shapes and patterns. |
| **MaxPooling2D (2×2)** | Reduces the size again to speed up learning. |
| **Flatten** | Turns the 2D feature maps into a 1D list to feed into dense layers. |
| **Dense (128, ReLU)** | Learns how features combine to recognize digits. |
| **Output (10, Softmax)** | Gives the final prediction — which digit (0–9) the image represents. |

**In short:**  
The CNN learns step by step first detecting simple edges, then shapes, and finally full digits.


### 3️⃣ Model Training

- **Epochs: 5** → The model goes through the entire training dataset 5 times to learn the patterns better.  
- **Batch size: 64** → The data is split into groups of 64 images that are processed together before updating the model weights.  
- **Validation split: 10%** → 10% of the training data is set aside to test how well the model is learning during training (not used for weight updates).


---


## How to Run

1- Install Dependencies:
  ```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

2-Run :

  ```bash
python cnn.py
```

3- Model Evaluation :

<p align="center">
<img width="1859" height="847" alt="Capture_2025_11_23_23_19_12_521" src="https://github.com/user-attachments/assets/4cc1677b-856e-46ca-b85f-4c3e0b321514" />
</p>

- The figure above shows how the model’s **accuracy** and **loss** changed during training.

- On the **left**, the **Accuracy** plot shows that both the training and validation accuracy steadily increased over the 5 epochs.  
  This means the model kept learning and improving its ability to correctly classify digits.

- On the **right**, the **Loss** plot shows that both the training and validation loss decreased over time.  
  Lower loss indicates that the model’s predictions are getting closer to the correct labels.

- Overall, the curves show **strong and consistent learning**, with both training and validation metrics improving smoothly — a sign of a well-trained model without overfitting.

---

<p align="center">
<img width="1859" height="1075" alt="Capture_2025_11_23_23_19_25_346" src="https://github.com/user-attachments/assets/5a46f260-0ea7-402d-bde8-00eb9b8d4ca8" />
</p>


- The figure above shows a few test images from the MNIST dataset along with their predicted and true labels.

- Each image displays the model’s **prediction (`pred`)** and the **actual digit (`True`)**.
  
- As seen, the model correctly identifies all samples showing that it can accurately recognize handwritten digits with high confidence.

---


<p align="center">
<img width="1723" height="1011" alt="Capture_2025_11_23_23_19_34_1" src="https://github.com/user-attachments/assets/f32674f5-a11f-4932-a3c1-d0fc63e32dee" />
</p>


- The heatmap above summarizes the model’s performance for each digit (0–9) using three main metrics:

  - **Precision:** How many of the predicted digits were actually correct.  
  - **Recall:** How many of the actual digits were correctly identified by the model.  
  - **F1-score:** A balanced measure combining both precision and recall.

- All scores are around **0.98–1.00**, showing that the model performs almost perfectly across all classes, with very few or no misclassifications.

---


<p align="center">
<img width="1385" height="1224" alt="Capture_2025_11_23_23_19_46_140" src="https://github.com/user-attachments/assets/619a050e-104e-4978-bb48-dbfee191b054" />
</p>


- The confusion matrix above shows how well the model classified each digit (0–9).

  - Each row represents the **true label**, and each column shows the **predicted label**.  
  - The diagonal cells (from top-left to bottom-right) represent **correct predictions**  most values are very high here.  
  - The few off-diagonal numbers represent **minor misclassifications**, which are very low overall.

- This matrix confirms that the CNN performs extremely well, correctly identifying nearly all digits with minimal errors.

---


<p align="center">
<img width="1850" height="1291" alt="Capture_2025_11_23_23_19_54_34" src="https://github.com/user-attachments/assets/9ade006f-7d92-4267-8a64-57566bb217d9" />
</p>


- The ROC (Receiver Operating Characteristic) curves above show the model’s ability to distinguish between the 10 digit classes.

  - Each line represents one digit (0–9), and all curves are near the top-left corner.  
  - The **AUC = 1.00** for all classes indicates **perfect classification performance**, meaning the model can confidently separate each digit without confusion.

- This confirms the CNN’s excellent predictive power and reliability across all digit categories.

  
 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
