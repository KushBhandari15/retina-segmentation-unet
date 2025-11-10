import cv2
from UNet_3blocks import Image_Segmentation_U3
from UNet_2blocks import Image_Segmentation_U2
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path):
    """Function to preprocess the image to required format"""
    # Load the image from the path provided
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB format
    image = image / 255.0 # Normalize the image to [0,1] by dividing pixel by 255
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1).unsqueeze(0) # Get image tensor as model needs for predicting

    return image, image_tensor

def predict(model, image_tensor, threshold=0.5):
    """Function to predict the class of the image"""
    image_tensor = image_tensor.to(device)

    with torch.no_grad(): # No gradient
        pred = model(image_tensor) # Predict
        prediction = torch.sigmoid(pred) # Convert using sigmoid function [0,1]
        prediction = (prediction > threshold).float()

    prediction = prediction.squeeze().cpu().numpy()
    return prediction

def visualize_predictions(image_paths, mask_paths, model, save_path='predictions_demo.png', threshold=0.5):
    """Function to visualize the predictions. The function helps juxtaposes the ground truth and predictions.
    The function will also save the visualization to user provided path"""
    num_samples = len(image_paths)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load and preprocess
        original_img, img_tensor = preprocess_image(img_path)

        # Load ground truth mask
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # Get prediction
        pred_mask = predict(model, img_tensor, threshold)

        # Plot
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(gt_mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(pred_mask, cmap='gray')
        axes[idx, 2].set_title('Model Prediction', fontsize=14, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()


def demo(model):
    """Demo prediction"""
    print(f"Using device: {device}")

    # Select 4 test images
    test_image_dir = './Data/Data/test/image'
    test_mask_dir = './Data/Data/test/mask'

    all_images = sorted(os.listdir(test_image_dir))[:4]  # Take first 4 images

    image_paths = [os.path.join(test_image_dir, img) for img in all_images]
    mask_paths = [os.path.join(test_mask_dir, img) for img in all_images]

    # Visualize
    visualize_predictions(
        image_paths=image_paths,
        mask_paths=mask_paths,
        model=model,
        save_path='demo.png',
        threshold=0.5
    )

def test_model():

    while True:
        print("\nChoose a model to run: ")
        print("Enter 1 to choose U-Net with 2 layers")
        print("Enter 2 to choose U-Net with 3 layers")
        print("Enter 3 to Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            u2 = Image_Segmentation_U2()
            u2.model.load_state_dict(torch.load("./U2_net.pth", map_location=device))
            u2.test()
            demo(u2.model)

        elif choice == '2':
            u3 = Image_Segmentation_U3()
            u3.model.load_state_dict(torch.load("./U3_net.pth", map_location=device))
            u3.test()
            demo(u3.model)
        elif choice == '3':
            break
        else:
            print("Invalid choice")

test_model()