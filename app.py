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

# Cached state to speed up consecutive clicks on the same image
CACHED_IMAGE = None
CACHED_STATE = None

def clear_cache():
    global CACHED_IMAGE, CACHED_STATE
    CACHED_IMAGE = None
    CACHED_STATE = None
    print("Inference state cache cleared.")

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

def visualize_results(pil_image, masks, boxes, scores, description_prefix, threshold=0.15):
    if masks is None or len(masks) == 0:
        return pil_image, f"No objects were detected."
        
    img_np = np.array(pil_image)
    h, w, c = img_np.shape
    overlay = np.zeros_like(img_np, dtype=np.uint8)
    
    # Visual color palette
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128),
        (255, 0, 127), (127, 255, 0), (0, 127, 255)
    ]
    
    valid_instances = 0
    info_text = f"Successfully detected instance(s) {description_prefix}:\n\n"
    valid_boxes = []
    
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        score_val = float(score.item()) if torch.is_tensor(score) else float(score)
        if score_val < threshold:
            continue
            
        valid_instances += 1
        
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
            valid_boxes.append((box, color, valid_instances))
            box_info = f", Box: [x1={int(box[0])}, y1={int(box[1])}, x2={int(box[2])}, y2={int(box[3])}]"
            
        info_text += f"• Instance {valid_instances}: Confidence Score: {score_val:.3f}{box_info}\n"
        
    if valid_instances == 0:
        return pil_image, f"No objects passed the confidence threshold of {threshold:.2f}."
        
    # Blend the color overlays with the original image
    alpha = 0.4
    blended = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
    
    # Draw bounding boxes and text labels on the image
    result_pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(result_pil)
    
    for box, color, inst_id in valid_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
        draw.text((box[0] + 5, box[1] + 5), f"#{inst_id}", fill=(255, 255, 255))
        
    info_text = f"Found {valid_instances} valid instance(s) above threshold.\n\n" + info_text
    return result_pil, info_text

def run_inference(input_image, prompt_text, threshold):
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        return None, "Error: Please load the model first using the Load Model settings in the left column."
        
    if input_image is None:
        return None, "Error: Please upload or select an image."
        
    if not prompt_text or prompt_text.strip() == "":
        return None, "Error: Please enter a prompt."
        
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
        
        return visualize_results(
            pil_image, 
            masks, 
            boxes, 
            scores, 
            description_prefix=f"matching '{prompt_text}'", 
            threshold=threshold
        )
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return None, f"Error running SAM 3 inference: {str(e)}\n\nDetails:\n{err_msg}"

def interactive_click_segment(input_image, select_data: gr.SelectData):
    global MODEL, PROCESSOR, CACHED_IMAGE, CACHED_STATE
    if MODEL is None or PROCESSOR is None:
        return None, "Error: Please load the model first using the Load Model settings in the left column."
        
    if input_image is None:
        return None, "Error: Please upload or select an image."
        
    try:
        # Convert input_image to PIL if it's a numpy array
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = input_image
            
        x, y = select_data.index
        print(f"User clicked at coordinates: col (x)={x}, row (y)={y}")
        
        # Check if we can reuse the cached image state
        is_same_image = False
        if CACHED_IMAGE is not None and CACHED_STATE is not None:
            if CACHED_IMAGE.size == pil_image.size:
                # Fast downsampled pixel check to see if the image is identical
                arr_cached = np.array(CACHED_IMAGE.resize((32, 32)))
                arr_current = np.array(pil_image.resize((32, 32)))
                diff = np.mean(np.abs(arr_cached - arr_current))
                if diff < 1.0:
                    is_same_image = True
                    
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if not is_same_image:
                        print("Encoding new image...")
                        CACHED_STATE = PROCESSOR.set_image(pil_image)
                        CACHED_IMAGE = pil_image
                    else:
                        print("Using cached image state.")
                        
                    # Build point coordinates and label tensors
                    point_coords = np.array([[x, y]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)
                    
                    # Predict raw instances
                    masks, scores, logits = MODEL.predict_inst(
                        CACHED_STATE,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=None,
                        multimask_output=True
                    )
            else:
                if not is_same_image:
                    print("Encoding new image...")
                    CACHED_STATE = PROCESSOR.set_image(pil_image)
                    CACHED_IMAGE = pil_image
                else:
                    print("Using cached image state.")
                    
                point_coords = np.array([[x, y]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                
                masks, scores, logits = MODEL.predict_inst(
                    CACHED_STATE,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True
                )
                
        # Draw mask and click dot
        if masks is None or len(masks) == 0:
            return pil_image, "No masks returned for this click point."
            
        # Select best candidate mask based on confidence score
        best_idx = torch.argmax(scores) if torch.is_tensor(scores) else np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        best_score_val = float(best_score.item()) if torch.is_tensor(best_score) else float(best_score)
        
        img_np = np.array(pil_image)
        overlay = np.zeros_like(img_np, dtype=np.uint8)
        
        if torch.is_tensor(best_mask):
            mask_np = best_mask.cpu().numpy()
        else:
            mask_np = np.array(best_mask)
            
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze(0)
            
        # Draw mask overlay in blue
        color = (0, 128, 255)
        overlay[mask_np > 0] = color
        
        alpha = 0.4
        blended = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
        
        result_pil = Image.fromarray(blended)
        draw = ImageDraw.Draw(result_pil)
        
        # Draw a green dot at the click coordinates
        dot_radius = 5
        draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=(0, 255, 0), outline=(255, 255, 255), width=2)
        
        info_text = f"Segmented object at coordinates ({x}, {y}):\n\n• Best Mask Confidence Score: {best_score_val:.3f}"
        return result_pil, info_text
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return None, f"Error running raw click prediction: {str(e)}\n\nDetails:\n{err_msg}"

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
        SAM 3 introduces **Promptable Concept Segmentation (PCS)** alongside native geometric point-based prediction.
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
            gr.Markdown("### 🖼️ 2. Inference Options")
            
            # Setup Tabs for different modes
            with gr.Tabs():
                with gr.TabItem("Raw Interactive Points"):
                    gr.Markdown(
                        """
                        ### 👆 Click-to-Segment (Raw Mode)
                        Click anywhere directly on the **Input Image** below to segment the object at that exact location. No text or keywords required!
                        """
                    )
                    with gr.Row():
                        # Set interactive=True to enable click selection events
                        raw_input_img = gr.Image(label="Input Image (Click here!)", type="pil", interactive=True)
                        raw_output_img = gr.Image(label="Segmented Output", type="pil", interactive=False)
                    
                    raw_detection_info = gr.Textbox(
                        label="Inference Details",
                        placeholder="Click coordinate logs and mask scores will appear here...",
                        interactive=False,
                        lines=3
                    )
                    
                    # Bind coordinates select click
                    raw_input_img.select(
                        fn=interactive_click_segment,
                        inputs=[raw_input_img],
                        outputs=[raw_output_img, raw_detection_info]
                    )
                    
                    # Clear cache when image is changed
                    raw_input_img.change(
                        fn=clear_cache,
                        inputs=[],
                        outputs=[]
                    )

                with gr.TabItem("Concept Prompting"):
                    gr.Markdown("Segment objects by typing a specific word or phrase.")
                    with gr.Row():
                        input_img = gr.Image(label="Input Image", type="pil")
                        output_img = gr.Image(label="Segmented Output", type="pil", interactive=False)
                    
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Text Prompt (Concept)",
                            placeholder="e.g. laptop, coffee mug, cat, person",
                            value="laptop",
                            info="Type what you want to segment"
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.15,
                            step=0.05,
                            label="Confidence Threshold",
                            info="Lower values show more segments; higher values filter out uncertain segments."
                        )
                    
                    segment_btn = gr.Button("🔮 Run Segmentation", variant="secondary")
                    detection_info = gr.Textbox(
                        label="Detection Details",
                        placeholder="Detection logs and scores will appear here...",
                        interactive=False,
                        lines=5
                    )
                    
                    segment_btn.click(
                        fn=run_inference,
                        inputs=[input_img, prompt, threshold_slider],
                        outputs=[output_img, detection_info]
                    )
                    
                    input_img.change(
                        fn=clear_cache,
                        inputs=[],
                        outputs=[]
                    )
                    
                with gr.TabItem("Auto-Segment Everything"):
                    gr.Markdown("Automatically find and segment everything in the photo using a generic open-vocabulary search.")
                    with gr.Row():
                        auto_input_img = gr.Image(label="Input Image", type="pil")
                        auto_output_img = gr.Image(label="Segmented Output", type="pil", interactive=False)
                    
                    with gr.Row():
                        auto_query = gr.Textbox(
                            label="Generic Search Query",
                            value="object",
                            info="Search term used to scan the image for general elements."
                        )
                        auto_threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.20,
                            step=0.05,
                            label="Confidence Threshold",
                            info="Lower values catch more background items; higher values restrict to main objects."
                        )
                        
                    auto_segment_btn = gr.Button("🔍 Auto-Segment Everything", variant="secondary")
                    auto_detection_info = gr.Textbox(
                        label="Detection Details",
                        placeholder="Detection logs and scores will appear here...",
                        interactive=False,
                        lines=5
                    )
                    
                    auto_segment_btn.click(
                        fn=run_inference,
                        inputs=[auto_input_img, auto_query, auto_threshold_slider],
                        outputs=[auto_output_img, auto_detection_info]
                    )
                    
                    auto_input_img.change(
                        fn=clear_cache,
                        inputs=[],
                        outputs=[]
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

    gr.Markdown(
        """
        ---
        ### 📖 Quick Guide
        1. **Check Status:** Check the **Model Status** box. By default, the app automatically loads the weights from the public mirror on start.
        2. **Raw Interactive Points:** Go to the **Raw Interactive Points** tab, upload an image, and click directly on the image to segment that item instantly!
        3. **Concept Prompting:** Upload an image, type a concept name, adjust the confidence slider, and click **Run Segmentation**.
        4. **Auto-Segment Everything:** Go to the **Auto-Segment Everything** tab, upload an image, and click **Auto-Segment Everything** to isolate all items.
        """
    )

if __name__ == "__main__":
    # Launch server with share=True to generate the public hotlink webpage
    demo.launch(share=True, debug=True)
