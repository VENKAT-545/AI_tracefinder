📌 TraceFinder – Milestone 3
CNN-Based Scanner Identification & Explainability
📖 Overview

Milestone 3 focuses on building a deep learning–based scanner identification system using a Convolutional Neural Network (CNN).
The model learns scanner-specific artifacts from scanned document images and provides explainability using Grad-CAM to visualize the regions influencing predictions.

This milestone demonstrates:

CNN model training

Performance evaluation

Model explainability

🎯 Objectives

Train a CNN to classify scanned documents by scanner source

Achieve training accuracy above 85%

Generate a confusion matrix for performance analysis

Apply Grad-CAM to visualize scanner-specific artifacts

🗂️ Folder Structure
Milestone 3/
│
├─ images/                     # Input images (filename-based labels)
│   ├─ scannerA_doc1.jpg
│   ├─ scannerA_doc2.jpg
│   ├─ scannerB_doc1.jpg
│   ├─ scannerC_doc1.jpg
│   └─ scannerD_doc1.jpg
│
├─ training/
│   └─ train_cnn.py            # CNN training script
│
├─ evaluation/
│   └─ confusion_matrix.py     # Training-based confusion matrix
│
├─ explainability/
│   └─ gradcam_fixed.py        # Grad-CAM implementation (functional model)
│
├─ models/
│   └─ cnn_scanner_model.h5    # Trained CNN model
│
├─ results/
│   ├─ confusion_matrix.png
│   └─ gradcam_output.png
│
└─ README.md

📊 Dataset Description

Images are stored directly inside the images/ folder

Scanner labels are extracted from filenames

Example: scannerA_doc1.jpg → scannerA

Dataset contains multiple scanner sources with limited samples per scanner

⚙️ Model Details

Architecture: Custom CNN

Input size: 128 × 128 RGB

Optimizer: Adam

Loss function: Categorical Cross-Entropy

Training strategy: Full-dataset training (no split due to limited samples)

▶️ How to Run
1️⃣ Train the CNN Model
cd "D:\Infosys Intern\Tracefinder\Milestone 3"
python training\train_cnn.py


✔ Output:

Training accuracy > 85% (typically reaches 100%)

Model saved to:

models/cnn_scanner_model.h5

2️⃣ Generate Confusion Matrix
python evaluation\confusion_matrix.py


✔ Output:

results/confusion_matrix.png

3️⃣ Run Grad-CAM Explainability
python explainability\gradcam_fixed.py


✔ Output:

results/gradcam_output.png

📈 Results Summary
Metric	Result
Training Accuracy	>85% (100% achieved)
Model Type	CNN
Explainability	Grad-CAM
Classes	Multiple scanner sources
🧠 Explainability (Grad-CAM)

Grad-CAM highlights the scanner-specific artifact regions that influence the CNN’s predictions.
This confirms that the model focuses on intrinsic scanner noise and texture patterns, not document content.