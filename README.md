# Handwritten Digit Recognition Using CNN

## Overview
This project implements a **Handwritten Digit Recognition** system using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The model is built using **Keras** and **TensorFlow**, trained to classify digits (0-9). Additionally, the project includes a **Tkinter-based GUI** for real-time digit prediction and an evaluation script to test the trained model.

## Features
- **CNN-based Image Classification**: Uses a deep learning model to classify handwritten digits.
- **Graphical User Interface (GUI)**: Allows users to draw digits and get real-time predictions.
- **Model Evaluation Script**: Tests the trained model on new input data.
- **Model Persistence**: Saves the trained model (`mnist.h5`) for future use.

---

## Project Structure
```
📂 Handwritten-Digit-Recognition
│── 📄 create_model.py         # Trains the CNN model on the MNIST dataset
│── 📄 evaluate_model.py       # Loads and tests the saved model
│── 📄 tkinter_gui.py          # GUI interface for digit prediction
│── 📄 mnist.h5                # Saved trained model
│── 📄 README.md               # Documentation
```

---

## Installation & Setup

### Prerequisites
Ensure you have **Python 3.x** installed along with the required dependencies.

### Install Dependencies
Run the following command to install the required libraries:
```sh
pip install tensorflow keras numpy matplotlib opencv-python pillow
```

---

## Model Training (`train_model.py`)
### Description
This script trains a **CNN model** on the MNIST dataset and saves it as `mnist.h5`.

### CNN Architecture
- **Two Convolutional Layers** (32 & 64 filters, ReLU activation)
- **MaxPooling Layer** (Reduces feature map size)
- **Dropout Layers** (Prevents overfitting)
- **Fully Connected Dense Layers** (256 neurons with ReLU, output layer with Softmax for classification)

### Run Training Script
```sh
python create_model.py
```

After training, the model is saved as `mnist.h5`.

---

## Model Evaluation (`evaluate_model.py`)
### Description
This script loads the saved model and evaluates its performance on the MNIST test dataset.

### Run Evaluation Script
```sh
python evaluate_model.py
```

It will output the accuracy of the model on test data.

---

## Graphical User Interface (`gui.py`)
### Description
A **Tkinter-based GUI** allows users to draw a digit using the mouse, and the trained model predicts the digit in real time.

### Run the GUI
```sh
python tkinter_gui.py
```

### Usage
1. A canvas appears where you can draw a digit.
2. Click **Predict** to get the classification result.
3. The predicted digit is displayed on the screen.

---

## Future Improvements
- Add support for real-time webcam-based digit recognition.
- Improve the model using deeper CNN architectures.
- Deploy the model using **Flask** or **FastAPI** for a web-based application.

---

## License
This project is licensed under the MIT License.

## Contributing
Feel free to contribute by submitting issues or pull requests.

