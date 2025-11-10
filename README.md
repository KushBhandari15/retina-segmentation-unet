# 🩸 Medical Image Segmentation (Retina Blood Vessel Detection)

## 📘 Overview
This project implements a **U-Net architecture from scratch** to perform **retinal blood vessel segmentation** on medical fundus images.  
The goal is to accurately identify and segment blood vessels in retinal images to assist in the early detection of retinal diseases such as diabetic retinopathy and macular degeneration.

This project focuses on understanding:
- How **Convolutional Neural Networks (CNNs)** work and are applied to image segmentation.
- How to **train/test segmentation models**.
- How to compute **Dice Score** and **Intersection over Union (IoU)** for performance evaluation.

---

## ⚙️ Dependencies
Install the required packages using:

## 📂 File Directory
File / Folder	Description
demo.py	Contains the main driver code to run and test the two models. The test_model() function provides both quantitative metrics and visual comparisons between model predictions and ground truth masks.
RetinaDataset.py	Contains a dataset class that loads and normalizes images and masks for training and testing.
helper.py	Includes custom loss functions (Dice, Focal, Combined) and metric computation functions (Dice, IoU, Accuracy).
UNet_2blocks.py	Defines the 2-block custom U-Net model architecture, along with its training, validation, and testing functions.
UNet_3blocks.py	Defines the 3-block custom U-Net model architecture, with deeper layers for improved feature extraction.
U2_net.pth / U3_net.pth	Pre-trained model weights for the 2-block and 3-block U-Net networks respectively.
image/	Contains output images, visualizations, and training iteration plots.


## 🚀 How to Run
🧩 1. Clone the Repository

First, clone this repository to your local machine:

- git clone https://github.com/username/repo-name.git
- cd repo-name 
- (Replace <your-username> and <your-repo-name> with your actual GitHub details.)

🧠 2. Install Dependencies

Install all required libraries:

- pip install -r requirements.txt

🧪 3. Run the Demo

Run the demo file to evaluate and visualize the models:

- python demo.py


The test_model() function will:

- Display a menu allowing you to select one of the available trained models (2-block or 3-block U-Net).
- Load the corresponding saved model weights.
- Evaluate the model on the test dataset.
- Print key metrics including Dice Score, IoU, and Accuracy.
- Visualize the predictions on the first 4 test images, comparing them to their ground truth masks.

## 🧮 Evaluation Metrics

- Dice Score: Measures overlap between predicted and ground truth regions.
- IoU (Intersection over Union): Quantifies similarity between predicted and true segmentation areas.
- Accuracy:	Measures the percentage of correctly classified pixels.

## 💾 Model Files

- U2_net.pth → Saved weights of the 2-block U-Net model.
- U3_net.pth → Saved weights of the 3-block U-Net model.

These models can be loaded in demo.py to perform segmentation testing and visualization.

## 📊 Notes

- The dataset consists of high-resolution retinal fundus images with binary masks (vessels = 1, background = 0).
- Images are normalized to [0, 1] and masks are thresholded to binary form.
- GPU acceleration is recommended to speed up training.
