# 🧬 Acute Lymphoblastic Leukemia (ALL) Image Classification using CNN (ResNet50)

This project focuses on building a deep learning model to classify images for the detection of **Acute Lymphoblastic Leukemia (ALL)** using a convolutional neural network (CNN) with a **pre-trained ResNet50** architecture. The ultimate goal is to assist in early diagnosis by automating the classification of peripheral blood smear (PBS) images.

---

## 📌 Project Overview

- **Domain:** Medical Imaging / Hematology  
- **Objective:** Classify PBS images into multiple classes to detect signs of ALL.  
- **Approach:** Transfer Learning using a pre-trained ResNet50 model with a custom classification head.  
- **Tools Used:** Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn

---

## 🛠️ Technologies and Libraries

- `TensorFlow` & `Keras` – for model creation and training  
- `Pandas` & `NumPy` – for data handling and preprocessing  
- `Matplotlib` & `Seaborn` – for visualizing dataset distribution and model performance  
- `ResNet50` – pre-trained model used as the base of the CNN  
- `OS` – for handling local file operations

---

## 🔁 ETL Process

A standard **ETL (Extract, Transform, Load)** pipeline was followed:

- **Extract**: Retrieved and validated the dataset from local directories.
- **Transform**: Divided the dataset into training, validation, and testing subsets. Ensured class balance across all subsets.
- **Load**: Loaded preprocessed image data directly into the model pipeline using `image_dataset_from_directory`.

---

## 📊 Data Visualization

- A bar chart visualized the **distribution of dataset subsets** (Train, Validation, Test).
- A pie chart illustrated the **proportion of data used** for each subset.
- A grouped bar chart showed **class distribution across subsets**, ensuring no class imbalance.

---

## 🧠 Model Architecture

- **Base:** Pre-trained ResNet50 (with top layers removed and weights frozen).
- **Input Shape:** 224 × 224 × 3
- **Custom Head:**  
  - `Flatten()`  
  - `Dropout(0.5)` – to reduce overfitting  
  - `Dense(4, activation='softmax')` – for multi-class classification

The model was compiled using:
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 📉 Training and Evaluation

- **Loss Function:** Categorical Crossentropy – suited for multi-class problems.
- **Optimizer:** Adam – combines advantages of RMSProp and momentum.
- **Evaluation Metrics:** Accuracy, Validation Loss

Training progress was tracked using training and validation loss curves to identify potential overfitting and underfitting.

---

## 📈 Future Work

- **Expand the dataset** to include more diverse images with varying characteristics.
- **Validate** the model with real-world clinical data.
- **Deploy** the model in low-resource environments as a decision-support tool for early diagnosis.

---

## ✅ Conclusion

This project demonstrates the potential of deep learning and transfer learning in medical diagnostics. With further validation and testing, it could serve as an effective aid for healthcare professionals, especially in areas with limited access to hematology experts.

---

## 📁 Dataset

The dataset consists of categorized blood smear images and was organized using directory-based labeling for compatibility with TensorFlow's image loader. Labeling was done by specialists using **flow cytometry**, ensuring high reliability.

---

## 📬 Contact

If you're interested in collaborating or learning more about this project, feel free to connect with me on [LinkedIn](https://www.linkedin.com) or send me a message!
