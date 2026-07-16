# Meta SAM 3 to 3D Model Generator

This project is a single-image to 3D model generator that combines **Meta Segment Anything Model 3 (SAM 3)** and **TripoSR**.

---

## 🎨 Features
1. **Interactive Segmentation**: Supports click-to-segment, concept prompting, and auto-segmentation modes to isolate objects.
2. **Transparent Masking**: Extracts the segmented object and replaces the background with a transparent alpha channel.
3. **Instant 3D Reconstruction**: Generates a textured 3D mesh model (.obj) from the isolated foreground in under a second using TripoSR.
4. **Interactive 3D Viewer**: Allows you to rotate, zoom, and download the `.obj` file directly within the browser interface.

---

## 🚀 Getting Started on Google Colab
To run this application, use the provided [sam3d_colab.ipynb](sam3d_colab.ipynb) notebook on a GPU instance (T4 or A100):

1. **Pull files**:
   ```python
   !git clone https://github.com/pluto-hiss/sam /content/sam
   %cd /content/sam
   ```
2. **Install Dependencies**:
   ```python
   !pip install "setuptools<81"
   !pip install -r requirements.txt
   ```
3. **Clone TripoSR**:
   ```python
   !git clone https://github.com/VAST-AI-Research/TripoSR.git
   ```
4. **Run verification test**:
   ```python
   !python test_3d_pipeline.py
   ```
5. **Launch App**:
   ```python
   !python app.py
   ```
