import os
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

# Global model state
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_model(load_source="community", hf_token=None):
    global MODEL, PROCESSOR
    if MODEL is not None:
        return "SAM 3 Model already loaded!"
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        checkpoint_path = None
        
        if load_source == "community":
            print("Downloading SAM 3 weights from community mirror (1038lab/sam3)...")
            checkpoint_path = hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt")
            print(f"Weights downloaded to: {checkpoint_path}")
        else:
            # Official Hugging Face gated repository
            if hf_token:
                hf_token = hf_token.strip()
                try:
                    from huggingface_hub import login
                    login(token=hf_token)
                except Exception as e:
                    return f"Failed to login to Hugging Face: {str(e)}"
            print("Downloading SAM 3 weights from official repository (facebook/sam3)...")
            # The official build function downloads the default checkpoint from facebook/sam3
            
        print(f"Building SAM 3 model on {DEVICE}...")
        
        if checkpoint_path:
            MODEL = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
        else:
            MODEL = build_sam3_image_model()
            
        MODEL.to(DEVICE)
        MODEL.eval()
        
        PROCESSOR = Sam3Processor(MODEL)
        return f"SAM 3 model loaded successfully on {DEVICE.upper()}!"
    except Exception as e:
        import traceback
        err_details = traceback.format_exc()
        return (
            f"Error loading SAM 3 model: {str(e)}\n\n"
            f"Details:\n{err_details}\n\n"
            "Please ensure:\n"
            "1. You are running on a GPU instance if using CUDA.\n"
            "2. If using the official source, you have valid Hugging Face access to facebook/sam3."
        )

def segment_image(input_image, prompt_text):
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        return None, "Error: Please load the model first by clicking 'Load SAM 3 Model'."
        
    if input_image is None:
        return None, "Error: Please upload or select an image."
        
    if not prompt_text or prompt_text.strip() == "":
        return None, "Error: Please enter a text prompt describing the concept to segment."
        
    try:
        # Convert input_image to PIL if it's a numpy array
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = input_image
            
        # Run inference
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_state = PROCESSOR.set_image(pil_image)
                    output = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt_text.strip())
            else:
                inference_state = PROCESSOR.set_image(pil_image)
                output = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt_text.strip())
            
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        
        if masks is None or len(masks) == 0:
            return pil_image, f"No objects matching '{prompt_text}' were detected."
            
        img_np = np.array(pil_image)
        h, w, c = img_np.shape
        overlay = np.zeros_like(img_np, dtype=np.uint8)
        
        # Color palette for instances
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 255, 128),
            (255, 0, 127), (127, 255, 0), (0, 127, 255)
        ]
        
        info_text = f"Successfully detected {len(masks)} instance(s) matching '{prompt_text}':\n\n"
        
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            # Convert mask to numpy bool mask
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
                
            if len(mask_np.shape) == 3:
                mask_np = mask_np.squeeze(0)
                
            color = colors[idx % len(colors)]
            overlay[mask_np > 0] = color
            
            box_info = ""
            if boxes is not None and len(boxes) > idx:
                box = boxes[idx]
                if torch.is_tensor(box):
                    box = box.cpu().numpy()
                box_info = f", Box: [x1={int(box[0])}, y1={int(box[1])}, x2={int(box[2])}, y2={int(box[3])}]"
                
            info_text += f"• Instance {idx+1}: Confidence Score: {score:.3f}{box_info}\n"
            
        # Blend the color overlays with the original image
        alpha = 0.4
        blended = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
        
        # Draw bounding boxes and text labels on the image
        result_pil = Image.fromarray(blended)
        draw = ImageDraw.Draw(result_pil)
        
        if boxes is not None:
            for idx, box in enumerate(boxes):
                if torch.is_tensor(box):
                    box = box.cpu().numpy()
                color = colors[idx % len(colors)]
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
                draw.text((box[0] + 5, box[1] + 5), f"#{idx+1}", fill=(255, 255, 255))
                
        return result_pil, info_text
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return None, f"Error running SAM 3 inference: {str(e)}\n\nDetails:\n{err_msg}"

# Auto-initialize model from community mirror on startup
auto_load_status = ""
try:
    print("Auto-initializing SAM 3 model from community mirror...")
    auto_load_status = init_model(load_source="community")
    print(auto_load_status)
except Exception as e:
    auto_load_status = f"Auto-initialization failed: {str(e)}. Please try manual loading below."
    print(auto_load_status)

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🎨 Meta Segment Anything Model 3 (SAM 3) Demo
        Welcome to the **SAM 3** interactive web application! 
        SAM 3 introduces **Promptable Concept Segmentation (PCS)**, which allows you to segment and track all instances of a concept using natural language text prompts (e.g., "glasses", "guitar", "cat").
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 1. Load Model Settings")
            load_source_radio = gr.Radio(
                label="Weight Source",
                choices=["community", "official_gated"],
                value="community",
                info="'community' downloads public weights immediately; 'official_gated' requires Hugging Face approval and token."
            )
            
            hf_token_input = gr.Textbox(
                label="Hugging Face Read Access Token",
                placeholder="hf_... (only needed for official_gated source)",
                type="password",
                visible=False
            )
            
            load_btn = gr.Button("🚀 Load/Reload SAM 3 Model", variant="primary")
            load_status = gr.Textbox(
                label="Model Status",
                value=auto_load_status,
                interactive=False
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ 2. Segment Concept")
            with gr.Row():
                input_img = gr.Image(label="Input Image", type="pil")
                output_img = gr.Image(label="Segmented Output", type="pil", interactive=False)
                
            prompt = gr.Textbox(
                label="Text Prompt (Concept)",
                placeholder="e.g. laptop, coffee mug, cat, person",
                info="Type what you want to segment"
            )
            
            segment_btn = gr.Button("🔮 Run Segmentation", variant="secondary")
            detection_info = gr.Textbox(
                label="Detection Details",
                placeholder="Detection logs and scores will appear here...",
                interactive=False,
                lines=5
            )
            
    # Show/hide token field depending on selected weight source
    def update_visibility(source):
        return gr.update(visible=(source == "official_gated"))
        
    load_source_radio.change(
        fn=update_visibility,
        inputs=[load_source_radio],
        outputs=[hf_token_input]
    )
    
    # Connect UI components
    load_btn.click(
        fn=init_model,
        inputs=[load_source_radio, hf_token_input],
        outputs=[load_status]
    )
    
    segment_btn.click(
        fn=segment_image,
        inputs=[input_img, prompt],
        outputs=[output_img, detection_info]
    )

    gr.Markdown(
        """
        ---
        ### 📖 Quick Guide
        1. **Check Status:** Check the **Model Status** box. By default, the app automatically loads the weights from the public mirror on start.
        2. **Load Model:** If not loaded, or if switching source, click **Load/Reload SAM 3 Model** (takes a moment to download).
        3. **Segment:** Upload an image, input a text prompt like *shoes*, and click **Run Segmentation**.
        """
    )

if __name__ == "__main__":
    # Launch server with share=True to generate the public hotlink webpage
    demo.launch(share=True, debug=True)
