# Music Genre Classification

An end-to-end music genre classification system using machine learning and deep learning. Features include audio preprocessing, MFCC/chroma extraction, mel-spectrogram generation, and CNN-based modeling to accurately classify audio tracks into multiple genres.

## Overview

Music genre classification is an important task in Music Information Retrieval (MIR).  
This project explores two main approaches:

- **Classical Machine Learning** using handcrafted audio features  
- **Deep Learning (CNN)** using spectrogram images  

The goal is to predict the musical genre of a given audio sample with high accuracy.

---

## Dataset

Audio files are organized by genre:

```
dataset/
│── classical/
│── pop/
│── jazz/
│── rock/
│── metal/
└── ...
```

You may use datasets such as **GTZAN**, **FMA**, or a custom dataset with proper labeling. I used GTZAN Datasets for training and testing purposes.

---

## Workflow

### **1. Audio Preprocessing**
- Resampling (e.g., 22050 Hz)  
- Converting stereo to mono  
- Silence trimming  
- Normalization  
- Splitting long audio into fixed-length segments  

### **2. Feature Extraction**
Extracted features include:

- MFCCs  
- Chroma  
- Spectral Centroid  
- Spectral Bandwidth  
- Rolloff  
- Zero-Crossing Rate  
- Tempo  

For deep learning models, **mel-spectrograms** or **MFCC images** are generated.

---

## Modeling

### **Classical ML Models**
- Random Forest  
- SVM  
- K-Nearest Neighbors  
- Gradient Boosting

These models use numerical features extracted from audio. They can be used respectively but these don't give good accuracy, so we used a CNN model.

### **Deep Learning Model**
A **Convolutional Neural Network (CNN)** trained on:

- Mel-spectrogram images  
- MFCC image representations  

---

## Evaluation

Models are evaluated using:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Confusion Matrix**

A standard train/validation/test split ensures fair evaluation.

---

## Key Highlights

- Complete audio ML + DL pipeline  
- Feature engineering + spectrogram generation  
- Multiple classifier comparisons  
- Genre-wise performance results  
- Easily extendable for experimentation  

---

## Future Improvements

- Add RNN / LSTM / GRU architectures  
- Use transfer learning on audio spectrograms  
- Deploy classification model via Streamlit or Flask  
- Real-time audio genre prediction  
- Expand dataset to improve robustness  

---

## Acknowledgements

- GTZAN / FMA dataset creators  
- Librosa for audio processing  
- TensorFlow, Keras, Scikit-Learn  

This project is licensed under the **MIT License**.

