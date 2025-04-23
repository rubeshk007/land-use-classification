# 📷 Land Use Classification using CNN & ResNet18

This project focuses on classifying aerial land use images using a custom Convolutional Neural Network (CNN) and a pretrained ResNet18 model. Built on the UCMerced LandUse dataset, the objective is multi-class image classification based on visual scene patterns captured in satellite imagery.

The pipeline spans from dataset preparation and augmentation to training and model evaluation using standard metrics.

---

## 🚀 Key Features

- UCMerced LandUse image classification using PyTorch
- Custom CNN model built from scratch
- Fine-tuning using a pretrained ResNet18 model
- Dataset split into training, validation, and test sets
- Evaluation using confusion matrices and classification metrics

---

## 📁 Dataset

Dataset: [UCMerced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

> **Note:** Dataset (`UCMerced_LandUse.zip`) must be downloaded separately and extracted into the working directory.

---

## 🧪 Technologies Used

- Python (3.x)
- PyTorch
- torchvision
- numpy, seaborn, matplotlib
- scikit-learn

---

## 🧠 Model Architectures

### Custom CNN
- 3 Convolutional layers with ReLU and MaxPooling
- Fully connected layers with ReLU
- Output layer with softmax activation for 21 land use classes

### Pretrained ResNet18
- Pretrained on ImageNet
- Final fully connected layer replaced with custom classifier
- Frozen feature extractor, trained only final layer

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (Visualized as heatmaps)

---

## 📈 Visualizations

- Training loss and accuracy per epoch
- Confusion matrices for both CNN and ResNet18 models

---

## 🏁 Usage Instructions

1. Clone this repository
2. Download and unzip the UCMerced LandUse dataset in the working directory
3. Run the script in a PyTorch-supported environment (e.g., Jupyter Notebook or Colab)
4. Follow console logs and plots for training and evaluation

---

## 🤝 Contributions

Contributions are welcome. Feel free to submit issues or pull requests to enhance the project.

---

## 📜 License

This project is licensed under the MIT License.
