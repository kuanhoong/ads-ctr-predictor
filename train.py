
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from utils import generate_dummy_data
from data_preprocessing import get_data_loaders
from model_2d_cnn import CNN2D
from model_1d_cnn import CNN1D
from model_combined import CombinedModel
from evaluate import compute_saliency_map, save_saliency_map

def train_one_epoch(model, loader, optimizer, criterion, device, mode='2d'):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        
        if mode == '2d':
            imgs, _ = inputs
            imgs = imgs.to(device)
            outputs = model(imgs)
        elif mode == '1d':
            _, feats = inputs
            feats = feats.to(device)
            outputs = model(feats)
        elif mode == 'combined':
            imgs, feats = inputs
            imgs = imgs.to(device)
            feats = feats.to(device)
            outputs = model(imgs, feats)
            
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device, mode='2d'):
    model.eval()
    running_mse = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            labels = labels.to(device)
            
            if mode == '2d':
                imgs, _ = inputs
                imgs = imgs.to(device)
                outputs = model(imgs)
            elif mode == '1d':
                _, feats = inputs
                feats = feats.to(device)
                outputs = model(feats)
            elif mode == 'combined':
                imgs, feats = inputs
                imgs = imgs.to(device)
                feats = feats.to(device)
                outputs = model(imgs, feats)
                
            loss = criterion(outputs.squeeze(), labels.squeeze())
            running_mse += loss.item()
    
    mse = running_mse / len(loader)
    rmse = np.sqrt(mse)
    return mse, rmse

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 0. Data Gen
    images, metrics, labels = generate_dummy_data(100, 74) # Small for test
    train_loader, test_loader = get_data_loaders(images, metrics, labels, batch_size=16)
    
    criterion = nn.MSELoss()
    
    # --- Experiment: 1D CNN Kernel Sizes (K=2 to 13) ---
    print("\n--- Phase 3.2: 1D CNN Experiments (Kernel Sizes) ---")
    mse_results = []
    kernel_sizes = range(2, 14) # 2 to 13
    
    # Input dim = 74 (metrics) + 15 (global) = 89
    
    for k in kernel_sizes:
        print(f"Training 1D CNN with Kernel K={k}...")
        model = CNN1D(input_dim=89, kernel_size=k).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        for ep in range(3): # Short train for demo
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mode='1d')
            
        # Eval
        mse, rmse = evaluate(model, test_loader, criterion, device, mode='1d')
        print(f"  K={k} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        mse_results.append(mse)
        
    # Plot MSE vs Kernel
    # Phase 4.1 Plot
    plt.figure()
    plt.plot(kernel_sizes, mse_results, marker='o')
    plt.title("MSE vs Kernel Size (1D CNN)")
    plt.xlabel("Kernel Size")
    plt.ylabel("MSE")
    plt.savefig("mse_vs_kernel.png")
    print("Saved mse_vs_kernel.png")

    # --- Experiment: 2D CNN ---
    print("\n--- Phase 3: 2D CNN Training ---")
    model_2d = CNN2D().to(device)
    opt_2d = optim.Adam(model_2d.parameters(), lr=0.01)
    
    for ep in range(5):
        loss = train_one_epoch(model_2d, train_loader, opt_2d, criterion, device, mode='2d')
        print(f"  Epoch {ep+1}: Loss {loss:.4f}")
    
    # Save the model
    torch.save(model_2d.state_dict(), "model_2d.pth")
    print("Saved model_2d.pth")
        
    mse_2d, rmse_2d = evaluate(model_2d, test_loader, criterion, device, mode='2d')
    print(f"2D CNN -> MSE: {mse_2d:.4f}, RMSE: {rmse_2d:.4f}")
    
    # Saliency Map (Phase 4.1)
    print("Generating Saliency Map...")
    # Get one image
    img, _ = next(iter(test_loader)) # (Batch, ...)
    img_input = img[0][0].unsqueeze(0) # (1, C, H, W)
    # Original for overlay (needs permute back to HWC and *255)
    img_orig = img_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    img_orig = img_orig.astype(np.uint8)
    # Ensure RGB
    # img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR) # If using opencv write which expects BGR
    img_orig_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

    saliency = compute_saliency_map(model_2d, img_input, None, device)
    save_saliency_map(img_orig_bgr, saliency, "saliency_map.png")
    print("Saved saliency_map.png")

    # --- Experiment: Combined Model ---
    print("\n--- Phase 3: Combined Model Training ---")
    model_comb = CombinedModel(input_dim_1d=89).to(device)
    opt_comb = optim.Adam(model_comb.parameters(), lr=0.01)
    
    for ep in range(5):
        loss = train_one_epoch(model_comb, train_loader, opt_comb, criterion, device, mode='combined')
        print(f"  Epoch {ep+1}: Loss {loss:.4f}")
        
    mse_comb, rmse_comb = evaluate(model_comb, test_loader, criterion, device, mode='combined')
    print(f"Combined Results -> MSE: {mse_comb:.4f}, RMSE: {rmse_comb:.4f}")

if __name__ == "__main__":
    main()
