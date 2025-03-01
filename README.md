# ctscan_analysis
# CT Scan Classification

This project focuses on classifying CT scan images into different categories using a convolutional neural network (CNN) model. The dataset contains images from lung and colon CT scans, categorized into different sets. The model is trained using TensorFlow and Keras and evaluated using various metrics including confusion matrix.

## Project Structure

```
.
├── data
│   └── lung_colon_image_set
│       ├── colon_image_sets
│       │   ├── colon_aca
│       │   └── colon_n
│       └── lung_image_sets
│           ├── lung_aca
│           ├── lung_n
│           └── lung_scc
├── extract_and_train.py
├── model_evaluation.py
└── README.md
```

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Installation

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/saibalaji13/ct-scan-classification.git
cd ct-scan-classification
```

Install the required Python packages:

```sh
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Dataset

Place the dataset archive (`archive.zip`) in the `data` directory. The script will extract the dataset and prepare it for training and evaluation.

### Training the Model

Run the `extract_and_train.py` script to extract the dataset and train the model:

```sh
python extract_and_train.py
```

This script will:
- Extract the dataset from the archive.
- Preprocess the images using data augmentation.
- Train a CNN model on the dataset.
- Save the trained model as `ctscan_classification_model.h5`.
- Plot and save the training history.

### Evaluating the Model

Run the `model_evaluation.py` script to evaluate the trained model and generate the confusion matrix:

```sh
python model_evaluation.py
```

This script will:
- Load the saved model.
- Generate predictions on the validation dataset.
- Compute and display the confusion matrix.

## Results

The results of the training and evaluation, including the confusion matrix, will be displayed and saved as images in the project directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
