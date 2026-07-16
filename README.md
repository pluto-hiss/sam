# SAM 3 Google Colab Demo with Hotlink Webpage

This repository contains the files required to launch a web-based interactive demo of Meta's **Segment Anything Model 3 (SAM 3)** running on a Google Colab backend. The web interface is built using **Gradio** and accessed via a public shareable URL (`*.gradio.live`).

SAM 3 introduces **Promptable Concept Segmentation (PCS)**, allowing you to segment and track all instances of a concept using natural language text prompts (e.g., "yellow school bus", "dog", "coffee mug").

---

## ⚡ Instant Setup (No Hugging Face Wait!)

By default, this application utilizes a public, ungated **community mirror** (`1038lab/sam3`) to download the `sam3.pt` checkpoint on startup. 

**This means you do NOT need a Hugging Face token or official registration to use this demo.** Simply launch the app, and the weights will download automatically!

---

## 🚀 How to Run the Demo

### Step 1: Push This Code to GitHub
To easily load the files into Google Colab, initialize git and push this repository to GitHub:
```bash
git init
git add .
git commit -m "Initialize SAM 3 Colab project"

# Create a repository on GitHub, then add your remote and push:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Open and Run the Notebook in Google Colab
1. Upload the `sam3_colab_demo.ipynb` file from this project to Google Colab, or open it directly from your GitHub repository using Colab's GitHub importer.
2. **Enable GPU:** Go to **Runtime > Change runtime type** in the Colab menu and select **T4 GPU** (or any available GPU).
3. Update the `GIT_REPO_URL` variable in **Step 1** of the notebook to match your GitHub repository URL:
   ```python
   GIT_REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git"
   ```
4. Run the cells in order:
   - **Step 1:** Clones your GitHub repository and sets the working directory.
   - **Step 2:** Installs general dependencies like `gradio`, `huggingface_hub`, and OpenCV.
   - **Step 3:** Clones Meta's official `facebookresearch/sam3` repository and installs it.
   - **Step 4:** Launches the Gradio web server.

### Step 3: Get Your Webpage Hotlink
Once the final cell runs and starts the server, look at the output logs for the public Gradio URL:
```text
Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces.
```
1. Click the `*.gradio.live` link to open the web demo in a new browser tab.
2. In the web interface:
   - The model automatically downloads weights from the community mirror and loads. Once the status shows **Loaded**, you are ready!
   - Upload an image in the **Input Image** box.
   - Type a concept in the **Text Prompt** box (e.g. `cat` or `backpack`).
   - Click **Run Segmentation** to visualize the masks and bounding boxes overlaid on the image!

---

## 🔒 Optional: Using Official Gated Weights
If you prefer to use Meta's official weights:
1. Request access on [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3).
2. Generate a Read Token under Hugging Face **Settings > Access Tokens**.
3. In the Gradio app panel, change the **Weight Source** radio selection to **official_gated**, paste your token, and click **Load/Reload SAM 3 Model**.
