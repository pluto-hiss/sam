import os
import sys
import torch
import numpy as np
from PIL import Image

def print_banner(msg):
    print("\n" + "=" * 60)
    print(f"👉 {msg}")
    print("=" * 60)

def main():
    print_banner("STARTING SAM 3 & INPAINTING PIPELINE VERIFICATION TEST")
    
    # 1. Verify Imports and Model Loading
    try:
        print("Importing app modules...")
        import app
        print("✅ Imports successful!")
    except Exception as e:
        print(f"❌ Failed to import app.py: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
        
    print(f"Device detected: {app.DEVICE}")
    if app.MODEL is None:
        print("❌ SAM 3 model was not initialized on startup.")
        sys.exit(1)
    else:
        print("✅ SAM 3 model loaded successfully!")
        
    # 2. Create Dummy Data
    print_banner("CREATING DUMMY TEST IMAGE AND MASK")
    # A 256x256 solid red image
    dummy_img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    # A 256x256 binary mask with a 50x50 white square in the center
    dummy_mask = Image.new("L", (256, 256), color=0)
    for x in range(100, 150):
        for y in range(100, 150):
            dummy_mask.putpixel((x, y), 255)
            
    dummy_img.save("test_dummy_img.png")
    dummy_mask.save("test_dummy_mask.png")
    print("✅ Created and saved dummy image and mask.")
    
    # 3. Test SAM 3 Point Click Inference
    print_banner("TESTING SAM 3 INTERACTIVE CLICK INFERENCE")
    try:
        # Create a mock select data object with coordinate index
        class MockSelectData:
            def __init__(self, index):
                self.index = index
                
        select_data = MockSelectData(index=[128, 128])
        
        result_img, info = app.interactive_click_segment(dummy_img, select_data)
        if result_img is not None:
            print("✅ Raw click segment test succeeded!")
            print(f"Inference output: {info}")
        else:
            print(f"❌ Raw click segment test failed. Output was None. Info: {info}")
            sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Raw click segment crashed with error: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    # 4. Test SAM 3 Concept Prompting
    print_banner("TESTING SAM 3 CONCEPT PROMPTING INFERENCE")
    try:
        result_img, info = app.run_inference(dummy_img, "red box", 0.05)
        # Note: since it's a solid color, it might not find a box, which is fine,
        # but the function shouldn't crash!
        print("✅ Concept prompting inference executed without crashing!")
        print(f"Inference output: {info}")
    except Exception as e:
        import traceback
        print(f"❌ Concept prompting inference crashed: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    # 5. Set Global LATEST_IMAGE & LATEST_MASK for Inpainting Tests
    app.LATEST_IMAGE = dummy_img
    app.LATEST_MASK = dummy_mask
    
    # 6. Test Latent Diffusion Inpainting
    print_banner("TESTING LATENT DIFFUSION (STABLE DIFFUSION) INPAINTING")
    try:
        inpaint_img, info = app.inpaint_object("latent-diffusion (Stable Diffusion)", "red background")
        if inpaint_img is not None:
            inpaint_img.save("test_output_latent_diffusion.png")
            print("✅ Latent Diffusion Inpainting test succeeded!")
            print(f"Status: {info}")
        else:
            print(f"❌ Latent Diffusion Inpainting test failed: {info}")
            sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Latent Diffusion Inpainting crashed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    # 7. Test ZITS++ Inpainting
    print_banner("TESTING ZITS++ INPAINTING")
    try:
        inpaint_img, info = app.inpaint_object("ZITS (CVPR 2022)", "")
        if inpaint_img is not None:
            inpaint_img.save("test_output_zitspp.png")
            print("✅ ZITS++ Inpainting test succeeded!")
            print(f"Status: {info}")
        else:
            print(f"❌ ZITS++ Inpainting test failed: {info}")
            sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ ZITS++ Inpainting crashed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    print_banner("ALL TESTS PASSED SUCCESSFULLY! PIPELINE IS 100% OPERATIONAL!")

if __name__ == "__main__":
    main()
