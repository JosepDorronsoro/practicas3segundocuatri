# Necessary installations: pip install torch torchvision scikit-learn matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
# Assuming this is a local module, I've renamed the import for consistency
from visualization.visualize_clustering_comparison import visualize_clustering_comparison

# ==========================================
# 1. DATA LOADING: MNIST (Images)
# ==========================================
print("Downloading and loading MNIST dataset...")
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

dataloader = DataLoader(mnist_data, batch_size=256, shuffle=True)
dataloader_eval = DataLoader(mnist_data, batch_size=256, shuffle=False)

# Preliminary Visualization
print("Visualizing raw samples from the dataset...")
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i in range(10):
    image_tensor, label = mnist_data[i]
    axes[i].imshow(image_tensor.squeeze().numpy(), cmap='gray')
    axes[i].set_title(f"Real: {label}")
    axes[i].axis('off')
plt.suptitle("MNIST Image Samples (28x28 pixels)", fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 2. CONVOLUTIONAL MODEL (LATENT DIM = 2)
# ==========================================
class ConvAutoencoder2D(nn.Module):
    # KEY FEATURE: latent_dim defaults to 2 for direct visualization
    def __init__(self, latent_dim=2): 
        super(ConvAutoencoder2D, self).__init__()
        
        # TODO: Adjust architecture to ensure the bottleneck is exactly 2D use, Sequenti Conv2D, activations, flatten an d Linear layers to achieve this. The decoder should mirror the encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            # TODO
        )

        # TODO: The decoder should mirror the encoder, but in reverse order. Use ConvTranspose2d for upsampling.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            # TODO
        )
        self.decoder_conv = nn.Sequential(
            # TODO: Mirror the encoder's Conv2D layers with ConvTranspose2D, ensuring the output is (1, 28, 28)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_dec = self.decoder_fc(z).view(-1, 32, 7, 7)
        x_reconstructed = self.decoder_conv(x_dec)
        return x_reconstructed, z

def train_deep_clustering_2d(dataloader, n_clusters=10, epochs=15, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Force latent_dim=2
    model = ConvAutoencoder2D(latent_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device) 
            optimizer.zero_grad()
            reconstruction, _ = model(images)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Average MSE Loss: {total_loss/len(dataloader):.4f}")

    model.eval()
    embeddings = []
    real_labels = [] 
    
    print("Extracting 2D latent features...")
    with torch.no_grad():
        for images, labels in dataloader_eval:
            images = images.to(device)
            _, z = model(images)
            embeddings.append(z.cpu().numpy())
            real_labels.append(labels.numpy())
            
    embeddings = np.vstack(embeddings)
    real_labels = np.concatenate(real_labels)

    print("Clustering with K-Means directly in 2D space...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_pred = kmeans.fit_predict(embeddings)
    
    # Return predicted labels, embeddings, ground truth, and centroids for plotting
    return labels_pred, embeddings, real_labels, kmeans.cluster_centers_

# ==========================================
# 3. EXECUTION AND VISUALIZATION
# ==========================================
# Run the training (takes a few minutes depending on hardware)
labels_pred, embeddings_2d, labels_real, centroids = train_deep_clustering_2d(dataloader, n_clusters=10, epochs=15)

# Direct Plot (No PCA required)
plt.figure(figsize=(12, 8))

# Color by PREDICTION (what the model thinks each group is)
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_pred, cmap='tab10', s=2, alpha=0.5)

# Draw the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='K-Means Centroids')

plt.title('Pure 2D Latent Space: Unsupervised MNIST Clustering')
plt.xlabel('Latent Neuron 1 (X-axis)')
plt.ylabel('Latent Neuron 2 (Y-axis)')
plt.legend()

cbar = plt.colorbar(scatter)
cbar.set_label('Predicted Cluster ID')
plt.show()

# Detailed comparison visualization
n_samples = 500
visualize_clustering_comparison(
    X=embeddings_2d[:n_samples], 
    y_true=labels_real[:n_samples], 
    y_pred=labels_pred[:n_samples], 
    centroids=centroids
)