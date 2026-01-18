# Ads CTR Predictor

This application predicts the Click-Through Rate (CTR) of display image advertisements using a 2D Convolutional Neural Network (CNN). It allows users to upload multiple ad designs and recommends the one with the highest predicted performance.

## Based on Research
This implementation is based on the paper:

**Cite this paper:**
Jhinn, W.L., Hoong, P.K., Chua, HK. (2020). Combination of 1D CNN and 2D CNN to Evaluate the Attractiveness of Display Image Advertisement and CTR Prediction. In: Lee, R. (eds) Software Engineering, Artificial Intelligence, Networking and Parallel/Distributed Computing. SNPD 2019. Studies in Computational Intelligence, vol 850. Springer, Cham. https://doi.org/10.1007/978-3-030-26428-4_11

## Features
- **Feature Extraction**: Extracts global image features (Brightness, Saturation, Colorfulness, Naturalness, Grayscale).
- **2D CNN Model**: Analyzes image visual patterns to predict attractiveness.
- **Combined Model**: Fuses image features with click metrics (implementation included).
- **Web App**: Streamlit interface for easy comparison of ad designs.
- **Saliency Maps**: Visualizes which parts of the ad contribute to the score.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a requirements.txt, typicaldeps: torch, torchvision, streamlit, opencv-python, matplotlib, numpy, pillow)*

2.  **Run Training (Optional)**
    To retrain the model (uses dummy data generator by default):
    ```bash
    python train.py
    ```

3.  **Run Application**
    ```bash
    streamlit run app.py
    ```

## Usage
1.  Launch the app.
2.  Upload up to 3 different image ad designs.
3.  View the predicted CTR scores.
4.  See the recommendation for the best performing ad.
