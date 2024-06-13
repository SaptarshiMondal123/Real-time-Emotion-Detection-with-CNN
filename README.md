# Real-time-Emotion-Detection-with-CNN
This repository hosts code for a real-time emotion detection system using a Convolutional Neural Network (CNN). The CNN model classifies facial expressions into seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise. The model is implemented in TensorFlow and Keras, trained on the FER-2013 dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Training Your Own Model](#training-your-own-model)
  - [Running the Emotion Detection](#running-the-emotion-detection)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to build a real-time system that can detect emotions from live camera feed. The model is implemented using TensorFlow and Keras and trained on the FER-2013 dataset.

## Setup Instructions
1. **Clone the repository:**
    ```bash
    git clone https://github.com/SaptarshiMondal123/Real-time-Emotion-Detection-with-CNN.git
    cd Real-time-Emotion-Detection-with-CNN
    ```

2. **Install the required dependencies:**
    - Make sure you have Python installed. Install required libraries using:
    ```bash
    pip install -r requirements.txt
    ```
    - This project requires `opencv-python`, `numpy`, `tensorflow`, and `keras`.

3. **Download the haarcascade classifier:**
    - Download the haarcascade classifier (`haarcascade_frontalface_default.xml`) and place it in the project directory.

## Usage

### Training Your Own Model
To train your own emotion detection model:

1. Open the `train_model.ipynb` Jupyter Notebook in your Python environment.
2. Follow the instructions in the notebook to preprocess the data, build the CNN model, train it using the FER-2013 dataset, and save the trained model (`model.keras`).
3. Once trained, you can use the `model.keras` file for real-time emotion detection.

### Running the Emotion Detection
To run the emotion detection system using your webcam:

1. Open the `evaluate_model.py` script in your Python environment.
2. Adjust the path to `haarcascade_frontalface_default.xml` if it is located in a different directory.
3. Run the script. It will open a live camera feed window that detects faces and predicts emotions in real-time.
4. The predicted emotion label will be displayed on each detected face in the video feed.

Example:
```bash
python evaluate_model.py
```

## Acknowledgements

- FER-2013 Dataset: The model is trained on the Face Expression Recognition Dataset (FER-2013) created by M. Sambare.
- Open Source Libraries:
  - TensorFlow, Keras, and OpenCV communities for their contributions to deep learning and computer vision.
