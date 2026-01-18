
import numpy as np

def generate_dummy_data(num_samples=100, num_metrics=74):
    """
    Generates dummy data to simulate the private Facebook dataset.
    Returns:
        images: (num_samples, 144, 144, 3) uint8
        metrics: (num_samples, num_metrics) float
        labels: (num_samples, 1) float (CTR)
    """
    print(f"Generating {num_samples} dummy samples...")
    # Random images
    images = np.random.randint(0, 256, (num_samples, 144, 144, 3), dtype=np.uint8)
    
    # Random metrics (raw values, e.g., 0-1000)
    metrics = np.random.rand(num_samples, num_metrics) * 1000
    
    # Random CTR labels (0.0 to 0.05 typical for CTR, using 0-1 for regression simplicity)
    labels = np.random.rand(num_samples, 1).astype(np.float32)
    
    return images, metrics, labels
