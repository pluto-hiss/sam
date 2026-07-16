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
    print_banner("STARTING SAM TO 3D PIPELINE VERIFICATION TEST")
    
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
    dummy_img = Image.new("RGB", (256, 256), color=(255, 0, 0))
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
        print(f"❌ Raw click segment crashed: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    # 4. Set Global LATEST_IMAGE & LATEST_MASK for 3D Generation Tests
    app.LATEST_IMAGE = dummy_img
    app.LATEST_MASK = dummy_mask
    
    # 5. Clone TripoSR if needed for testing (normally done in notebook, but let's clone if missing)
    if not os.path.exists("TripoSR"):
        print("TripoSR repo not found locally. Cloning VAST-AI-Research/TripoSR...")
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git"])
        
    # 6. Test TripoSR 3D Generation
    print_banner("TESTING TRIPOSR 3D MESH GENERATION")
    try:
        out_mesh_path, info = app.generate_3d_model()
        if out_mesh_path is not None and os.path.exists(out_mesh_path):
            print("✅ TripoSR 3D Mesh Generation test succeeded!")
            print(f"Output saved to: {out_mesh_path}")
            print(f"Status: {info}")
        else:
            print(f"❌ TripoSR 3D Mesh Generation failed: {info}")
            sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ TripoSR 3D Mesh Generation crashed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    print_banner("ALL TESTS PASSED SUCCESSFULLY! PIPELINE IS 100% OPERATIONAL!")

if __name__ == "__main__":
    main()
