# Breast Cancer Classification Using EfficientNetB0 and Resnet50 

This repository contains a project for classifying breast cancer using deep learning, specifically fine-tuning the pre-trained EfficientNetB0 model. The dataset used is for breast cancer image classification, and the model is trained in the provided Jupyter notebook.

## Project Overview

The goal of this project is to utilize transfer learning by fine-tuning the EfficientNetB0 model on a breast cancer dataset. The model aims to predict whether a given input image belongs to a class associated with breast cancer.

### Key Features
- **EfficientNetB0**: A state-of-the-art image classification model that is pre-trained on ImageNet.
- **Transfer Learning**: Only the final layers of the model are fine-tuned, allowing for faster convergence and improved accuracy on small datasets.
- **TensorFlow and Keras**: The model is built using TensorFlow 2.x and Keras, and can be easily run in Google Colab.

## File Structure

- `efficientnetB0_breast_cancer_colab.ipynb`: The main Jupyter notebook containing the code for loading the dataset, model training, and evaluation.
- `README.md`: The documentation for the repository (this file).

## Prerequisites

To run the notebook, the following libraries need to be installed:

- `tensorflow`
- `numpy`
- `matplotlib`
- `opencv-python`
- `pandas`
- `scikit-learn`

You can install them using:

```bash
pip install tensorflow numpy matplotlib opencv-python pandas scikit-learn
```

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/matt7salomon/breast_cancer_from_mammograms_deep_learning
   cd breast_cancer_from_mammograms_deep_learning
   ```

2. Open the Jupyter notebook:
   - Run the notebook in a local Jupyter environment or Google Colab.

3. Run all cells in the notebook:
   - The notebook will walk through loading the data, preprocessing, building the model, training, and evaluating the model.

## Model Architecture

The model used is **EfficientNetB0**, which is pre-trained on ImageNet. Only the final fully connected layers are trained on the breast cancer dataset. The final layer has been replaced with a dense layer with an output size corresponding to the number of classes in the dataset (e.g., binary classification).

### Loss Function and Metrics

- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: `Adam`
- **Evaluation Metrics**: `Accuracy`

## Dataset

The dataset used for this project should contain images related to breast cancer classification. Ensure that your dataset is organized into subfolders by class (e.g., `train/benign`, `train/malignant`, `val/benign`, `val/malignant`).

## Results

After training the model, it achieves competitive accuracy on the validation set. The model is suitable for deployment in environments where high accuracy in medical image classification is required.

## Contributing

Contributions are welcome! Please open a pull request or issue if you would like to contribute code or report bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
