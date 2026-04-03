# 🎯 Emoji Object Localization & Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/141CAQ6woRpxZcV25BcfHjW5ODCJD20I-?usp=sharing)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

A custom Multi-Task Convolutional Neural Network (CNN) built from scratch to simultaneously classify emojis and predict their exact bounding box coordinates on a generated canvas.

## 🚀 Project Overview

This project demonstrates the core concepts of **Multi-Output Deep Learning** and **Computer Vision**. Instead of simply predicting *what* an object is, the model also predicts *where* it is. 

The network takes a `144x144` RGB image containing a single emoji and processes it through a shared feature extractor before splitting into two distinct heads:
1.  **Classification Head:** Identifies the emoji out of 9 possible distinct classes.
2.  **Regression Head:** Predicts the `(x, y)` coordinates for a `52x52` bounding box.

### Visual Results
<br>
<img src="assets/demo.png" alt="Model Demo" width="800"/>

---

## 🧠 Technical Architecture

The model was built using the **TensorFlow/Keras Functional API** to support branching output layers.

* **Shared Feature Extractor:** 5 sequential Convolutional blocks. Each block consists of `Conv2D` -> `BatchNormalization` -> `MaxPool2D`. Filter sizes scale up intelligently from 16 to 256 to capture complex hierarchical features.
* **Classification Output:** Flattened vector fed into a Dense layer with `Softmax` activation. Optimized via `Categorical Crossentropy`.
* **Bounding Box Output:** Flattened vector fed into a Dense layer with linear activation. Optimized via `Mean Squared Error (MSE)`.

---

## 📐 Custom Evaluation Metric (IoU)

To accurately track the model's localization performance during training, I engineered a custom **Intersection over Union (IoU)** metric class. 
* This calculates the exact pixel-overlap ratio between the predicted bounding box and the ground truth.
* It ensures the model is evaluated on physical, visual accuracy rather than relying solely on the abstract MSE loss number.

---

## ⚙️ Advanced Training Pipeline

* **Custom Data Generator:** Yields infinite batches of dynamically generated emoji canvases (`while True` loop). This prevents RAM overflow and allows for continuous training data.
* **Learning Rate Scheduler:** Implements a step decay strategy (reduces learning rate by 80% every 5 epochs) to ensure stable convergence and prevent gradient overshooting.
* **Early Stopping:** Actively monitors the custom `box_out_iou` metric, halting training if the localization accuracy stagnates for 3 consecutive epochs.
* **Custom Callbacks:** Uses Keras callbacks to visualize model predictions and draw comparative bounding boxes on unseen test data at the end of every single epoch.

---

## 💻 Getting Started (Local Setup)

Want to run this locally? Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/emoji-localization.git](https://github.com/YOUR_GITHUB_USERNAME/emoji-localization.git)
   cd emoji-localization
   ```
2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook:**
   Open the Jupyter Notebook to explore the architecture, train the model, and visualize the bounding boxes.
   ```bash
   jupyter notebook notebooks/emoji_localization.ipynb
   ```
   
---

## 📁 Repository Structure
```
emoji-localization/
│
├── notebooks/
│   └── emoji_localization.ipynb       # Core training and architecture code
│
├── assets/
│   └── demo.png                       # Output visualizations
│
├── .gitignore                         # Standard Python/Keras gitignore
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```
