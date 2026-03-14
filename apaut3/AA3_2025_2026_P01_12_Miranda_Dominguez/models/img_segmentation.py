import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import warnings

def get_sample_image(img_name="china.jpg"):
    """Loads a sample image built into scikit-learn.
    
    Returns:
        numpy.ndarray: RGB image in (Height, Width, 3) format with values between 0 and 1.
    """
    # Using 'flower.jpg' or 'china.jpg'
    # Ignore deprecation warnings from sklearn imports if they occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = load_sample_image(img_name)

    # Normalize pixel values to [0, 1] for better K-means convergence
    image = np.array(image, dtype=np.float64) / 255.0
    return image

def segment_image(image, k=4):
    """Applies K-means to image pixels to reduce its color palette (Color Quantization).

    Args:
        image (numpy.ndarray): Original image (Height, Width, 3).
        k (int): Number of desired colors/clusters.

    Returns:
        numpy.ndarray: Segmented image with the same original shape.
    """
    height, width, channels = image.shape
    
    # 1. Flatten the image: from (Height, Width, 3) to (Pixels, 3)
    pixel_data = np.reshape(image, (height * width, channels))
    
    # 2. Train a class method and predict cluster labels for each pixel
    # Using n_init='auto' for speed in this visual demonstration
    
    clustering = KMeans(n_init='auto')
    
    labels = clustering.fit_predict(pixel_data)
    
    # 3. Retrieve the centroid colors
    centroids = clustering.cluster_centers_
    
    # 4. Replace each pixel with the color of its assigned centroid
    segmented_pixels = centroids[labels]
    
    # 5. Reshape back to the original form (Height, Width, 3)
    segmented_image = np.reshape(segmented_pixels, (height, width, channels))
    
    # Ensure values are within the valid [0, 1] range for matplotlib
    segmented_image = np.clip(segmented_image, 0, 1)
    
    return segmented_image

# ==========================================
# EXECUTION AND VISUALIZATION
# ==========================================

for img_name in ["flower.jpg", "china.jpg"]:
    print(f"Processing image: {img_name}")
    original_image = get_sample_image(img_name)

    # 2. Segment with different K values (colors)
    print("Training models...")
    image_k2 = segment_image(original_image, k=2)   # Binary segmentation (Background vs Foreground)
    image_k4 = segment_image(original_image, k=4)   # 4 dominant colors
    image_k16 = segment_image(original_image, k=16) # Realistic compression (Color Quantization)

    # 3. Visualize results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original (Thousands of colors)")
    axes[0].axis('off')

    # K = 2
    axes[1].imshow(image_k2)
    axes[1].set_title("K-means (K=2)\nBackground/Foreground Mask")
    axes[1].axis('off')

    # K = 4
    axes[2].imshow(image_k4)
    axes[2].set_title("K-means (K=4)\nBasic Segmentation")
    axes[2].axis('off')

    # K = 16
    axes[3].imshow(image_k16)
    axes[3].set_title("K-means (K=16)\nColor Quantization (16 colors)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()