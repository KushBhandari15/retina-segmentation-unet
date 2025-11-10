import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import RetinaDataset as RD
from helper import compute_metrics, CombinedLoss

# Class for U Net with 3 blocks
class Image_Segmentation_U3:

    def __init__(self):
        # Setup training dataset
        training_dataset = RD.RetinaDataset(
            image_dir="./Data/Data/train/image", mask_dir="./Data/Data/train/mask"
        )
        # Splitting the dataset into - 80% training and 20% validation
        val_ratio = 0.2
        n_val = int(len(training_dataset) * val_ratio)
        n_train = len(training_dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            training_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        # Setup testing dataset
        self.test_ds = RD.RetinaDataset(
            image_dir="./Data/Data/test/image", mask_dir="./Data/Data/test/mask"
        )
        # Using dataloader to process large amount of data easily
        self.train_loader = DataLoader(self.train_ds, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=4, shuffle=False)
        # Using gpu to expedite the training process
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.setup_model()
        self.model.to(self.device)

        # Setup Loss function (Focal and Dice Loss combined)
        self.criterion = CombinedLoss(dice_weight=0.7, focal_weight=0.3) # Weightage 70% Dice Loss and 30% Focal Loss

        # Optimizer settings
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,  # Start with higher LR
            weight_decay=1e-5
        )
        #  Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            min_lr=1e-6
        )

    def setup_model(self):

        # UNet with 3 layer architecture
        class UNet3Blocks(nn.Module):
            def __init__(self, in_channels=3, base_filters=64, out_classes=1):
                super().__init__()
                f = base_filters

                # Each encoder and decoder level will have 2 convolution layers with ReLU activation function
                # Encoder 1
                self.encoder1 = nn.Sequential(
                    nn.Conv2d(in_channels, f, 3, padding=1),
                    nn.BatchNorm2d(f),  # Added BatchNorm
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f, f, 3, padding=1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True)
                )
                # Encoder 2
                self.encoder2 = nn.Sequential(
                    nn.Conv2d(f, f * 2, 3, padding=1),
                    nn.BatchNorm2d(f * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f * 2, f * 2, 3, padding=1),
                    nn.BatchNorm2d(f * 2),
                    nn.ReLU(inplace=True)
                )
                # Encoder 3
                self.encoder3 = nn.Sequential(
                    nn.Conv2d(f * 2, f * 4, 3, padding=1),
                    nn.BatchNorm2d(f * 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f * 4, f * 4, 3, padding=1),
                    nn.BatchNorm2d(f * 4),
                    nn.ReLU(inplace=True)
                )
                # Max Pool
                self.pool = nn.MaxPool2d(2)

                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(f * 4, f * 8, 3, padding=1),
                    nn.BatchNorm2d(f * 8),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f * 8, f * 8, 3, padding=1),
                    nn.BatchNorm2d(f * 8),
                    nn.ReLU(inplace=True)
                )

                # Decoder 3
                self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
                self.decoder3 = nn.Sequential(
                    nn.Conv2d(f * 8, f * 4, 3, padding=1),
                    nn.BatchNorm2d(f * 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f * 4, f * 4, 3, padding=1),
                    nn.BatchNorm2d(f * 4),
                    nn.ReLU(inplace=True)
                )

                # Decoder 2
                self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
                self.decoder2 = nn.Sequential(
                    nn.Conv2d(f * 4, f * 2, 3, padding=1),
                    nn.BatchNorm2d(f * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f * 2, f * 2, 3, padding=1),
                    nn.BatchNorm2d(f * 2),
                    nn.ReLU(inplace=True)
                )

                # Decoder 1
                self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
                self.decoder1 = nn.Sequential(
                    nn.Conv2d(f * 2, f, 3, padding=1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f, f, 3, padding=1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True)
                )

                # Final Convolution Layer
                self.final = nn.Conv2d(f, out_classes, 1)

            def forward(self, x):
                # Down-sampling and encoder
                e1 = self.encoder1(x); p1 = self.pool(e1)
                e2 = self.encoder2(p1); p2 = self.pool(e2)
                e3 = self.encoder3(p2); p3 = self.pool(e3)

                # Bottleneck
                b = self.bottleneck(p3)

                # Up-sampling and decoder
                u3 = self.up3(b); c3 = torch.cat([u3, e3], dim=1); d3 = self.decoder3(c3)
                u2 = self.up2(d3); c2 = torch.cat([u2, e2], dim=1); d2 = self.decoder2(c2)
                u1 = self.up1(d2); c1 = torch.cat([u1, e1], dim=1); d1 = self.decoder1(c1)

                # Final Layer
                out = self.final(d1)
                return out

        # Base filter = 64 gave best model performance
        model = UNet3Blocks(in_channels=3, base_filters=64, out_classes=1)
        return model

    def train(self, epochs=10):
        """Function to train the model. The model will be trained for user provided number of epochs.
        The function will also handle plateau by the help of learning rate scheduler"""
        print("--- Starting Training ---")
        # Iterate over given epochs
        for epoch in range(epochs):
            self.model.train() # Set to training
            running_loss = 0.0 # Store running loss

            for imgs, masks in self.train_loader: # Train loader provides image with its corresponding mask together
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks) # Loss criterion
                self.optimizer.zero_grad() # Clear gradient
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item() # Update loss

            epoch_loss = running_loss / len(self.train_loader) # Total Loss divided by number of batches
            val_loss = self.validate() # Call validation function
            print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss) # Learning rate scheduler

    def validate(self):
        """Function for validation of model. The model will be validated using 20% of the training dataset.
        The function will also compute and print metrics including IoU, Dice, and Accuracy"
        """
        # Set in evaluation mode
        self.model.eval()
        total_iou, total_dice, total_loss, total_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0 # Keep track of metrics
        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                val_preds = self.model(imgs) # Predict
                loss = self.criterion(val_preds, masks) # Compute loss
                iou, dice, acc = compute_metrics(val_preds, masks) # Calculate metrics: IoU, Dice, and Accuracy
                total_loss += loss.item(); total_iou += iou; total_dice += dice; total_acc += acc; n_batches += 1

        # Average metrics = Total Value/ Number of batches
        avg_iou = total_iou / n_batches
        avg_dice = total_dice / n_batches
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        print(f"Val loss: {avg_loss}; IoU: {avg_iou}; Dice: {avg_dice}; Accuracy: {avg_acc}")
        return avg_loss

    def test(self):
        """Function to test the model using testing data set.
        The function will compute and print metrics including IoU, Dice, Accuracy."""
        print("--- Starting Testing of unet3 ---")
        self.model.eval() # Set to evaluation mode
        total_iou, total_dice, total_loss, total_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0 # Store metrics
        with torch.no_grad():
            for imgs, masks in self.test_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                test_preds = self.model(imgs)
                loss = self.criterion(test_preds, masks)
                iou, dice, acc = compute_metrics(test_preds, masks) # Calculate metrics
                total_loss += loss.item(); total_iou += iou; total_dice += dice; total_acc += acc; n_batches += 1

        # Average metrics = Total Value/ Number of batches
        avg_iou = total_iou / n_batches
        avg_dice = total_dice / n_batches
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        print(f"Test loss: {avg_loss}; IoU: {avg_iou}; Dice: {avg_dice}; Accuracy: {avg_acc}")
        return avg_loss

    def save_model(self, path = "U3_net.pth"):
        """Function to save model state dictionary to user provided directory"""
        print("--- Saving Model ---")
        torch.save(self.model.state_dict(), path) # Saving model parameters
        print(f"Model state_dict saved to {path}")


def execute_all(epochs=10):
    """Helper function to execute all functions in this module"""
    helper = Image_Segmentation_U3()
    helper.train(epochs=epochs)
    helper.test()
    helper.save_model()

if __name__ == "__main__":
    execute_all(epochs=7)