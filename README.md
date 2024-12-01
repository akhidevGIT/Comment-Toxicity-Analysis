# Comment Toxicity Analysis Using TensorFlow and LSTM Model

## 📜 Overview

This project tackles the problem of **toxic comment classification**, where each comment is analyzed and classified into one or more of the following six categories:

1. **Toxic**
2. **Severe Toxic**
3. **Obscene**
4. **Threat**
5. **Insult**
6. **Identity Hate**

The solution employs a **Deep Learning model** built using TensorFlow's Sequential API. The model leverages an **Embedding layer**, **Bidirectional LSTM**, and **Dense layers** to perform multi-label classification on binary target labels.

---

## 🛠️ Features

- **Text Preprocessing**:
  - Used `TextVectorization` for converting raw text into sequences of token indices.
  - Converted the dataset into TensorFlow `tf.data.Dataset` for efficient batching and preprocessing.

- **Model Architecture**:
  - **Embedding Layer**: Converts tokens into dense embeddings.
  - **Bidirectional LSTM**: Captures both past and future context within the sequences.
  - **Dense Layers**: Outputs probabilities for each of the six target categories.

- **Evaluation Metrics**:
  - Precision, Recall, and Categorical Accuracy were used to evaluate the model.

---

## 🔧 Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow (Keras API)
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - TensorFlow Utilities: `tensorflow`, `tensorflow.keras`, `tensorflow.data`
  - Text Preprocessing: `tensorflow.keras.layers.TextVectorization`

---

## 📂 File Structure

```plaintext
Comment-Toxicity-Analysis/
│
├── data/                      # Dataset folder
│   ├── comments_toxicity.csv  # Training dataset
│
├── commentToxicity.ipynb      # Comment toxicity analysis Jupyter notebook
│
├── models/                    # Saved model and checkpoints
│   ├── comment_toxicity.h5    # Final trained model
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
