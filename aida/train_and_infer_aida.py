import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. AI DATA ASSIMILATION MODEL ARCHITECTURE (3D CNN / U-Net style)
# ==============================================================================
class Neural3DVar(nn.Module):
    """
    3D Convolutional Neural Network that maps 3D Background (x_b) and 3D Innovations
    to the 3D Analysis Increment (dx).
    """
    def __init__(self, in_channels=2, out_channels=1):
        super(Neural3DVar, self).__init__()
        
        # Encoder (Feature Extraction)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # Decoder (Predict Increment dx)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
            # No activation at output because increments can be positive or negative
        )

    def forward(self, x_b, innovations_grid):
        # Stack background and grid-mapped innovations along channel dimension
        # Input shape: (Batch, 2, Levels, Lat, Lon)
        x_in = torch.cat([x_b, innovations_grid], dim=1)
        
        feat1 = self.enc1(x_in)
        bottle = self.bottleneck(feat1)
        
        # Predict Increment dx
        dx = self.dec1(bottle)
        
        # Analysis State x_a = x_b + dx
        x_a = x_b + dx
        
        return x_a, dx


# ==============================================================================
# 2. TRAINING ROUTINE & CHECKPOINT SAVING
# ==============================================================================
def train_model(dataloader, num_epochs=10, checkpoint_path="aida_checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AI-DA Model on Device: {device}")
    
    # Model accepts 2 channels (x_b, innovation_grid) -> outputs 1 channel (dx)
    model = Neural3DVar(in_channels=2, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (x_b, innovación_grid, dx_target) in enumerate(dataloader):
            x_b = x_b.to(device)
            innovación_grid = innovación_grid.to(device)
            dx_target = dx_target.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            x_a_pred, dx_pred = model(x_b, innovación_grid)
            
            # Compute Loss on predicted increment vs target increment
            loss = criterion(dx_pred, dx_target)
            
            # Backward pass & update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

    # Save Checkpoint File
    checkpoint = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n[✓] Checkpoint saved successfully to: '{checkpoint_path}'")


# ==============================================================================
# 3. INFERENCE ROUTINE (USING CHECKPOINT FILE)
# ==============================================================================
def run_inference_from_checkpoint(checkpoint_path, x_b_tensor, innovation_tensor):
    """
    Loads checkpoint file, applies it to background x_b and innovations,
    and returns predicted x_a and dx in milliseconds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture and load weights
    model = Neural3DVar(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"\n[✓] Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Inference
    with torch.no_grad():
        x_b_input = x_b_tensor.to(device)
        inno_input = innovation_tensor.to(device)
        
        # Single forward pass
        x_a_pred, dx_pred = model(x_b_input, inno_input)

    return x_a_pred.cpu(), dx_pred.cpu()


# ==============================================================================
# 4. EXAMPLE DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    # Simulate batch dimensions: (Batch=4, Channels=1, Height=32, Lat=181, Lon=360)
    B, C, H, W, D = 4, 1, 32, 181, 360
    
    # Mock data tensors
    mock_x_b = torch.randn(B, C, H, W, D)
    mock_inno = torch.randn(B, C, H, W, D) * 0.5
    mock_dx_target = torch.randn(B, C, H, W, D) * 0.2

    # Create dummy dataset & dataloader
    dataset = torch.utils.data.TensorDataset(mock_x_b, mock_inno, mock_dx_target)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 1. Train and save checkpoint
    ckpt_file = "gfs_aida_model.pt"
    train_model(dataloader, num_epochs=3, checkpoint_path=ckpt_file)

    # 2. Load checkpoint and run inference on a new Xb
    new_x_b = torch.randn(1, 1, H, W, D)
    new_inno = torch.randn(1, 1, H, W, D) * 0.5

    x_a, dx = run_inference_from_checkpoint(ckpt_file, new_x_b, new_inno)

    print(f"\nInference Results:")
    print(f"Output x_a Shape: {x_a.shape}")
    print(f"Output dx  Shape: {dx.shape}")
    print(f"Max Predicted Increment: {dx.abs().max().item():.4f}")

