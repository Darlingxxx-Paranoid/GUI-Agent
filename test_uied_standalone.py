
import os
import cv2
import json
import logging
import numpy as np
import time
from Perception.uied.detect import WidgetDetector

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_uied():
    # 1. Configuration
    # Make sure this path exists on your machine or update it
    input_screenshot = r"data\screenshots\step_1_before.png" 
    output_dir = r"data\cv_output_standalone"
    
    if not os.path.exists(input_screenshot):
        logger.error(f"Input screenshot not found: {input_screenshot}")
        logger.info("Please ensure you have a screenshot at that path or update the 'input_screenshot' variable.")
        return

    # 2. Cleanup and Create Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # 3. Initialize WidgetDetector
    logger.info("Initializing WidgetDetector...")
    detector = WidgetDetector()
    detector.output_root = output_dir
    detector.resize_length = 800 # Resize height for detection

    # 4. Run Detection
    logger.info(f"Running detection on: {input_screenshot}")
    start_time = time.time()
    try:
        # detect() returns: img_res_path, resize_ratio, elements["compos"]
        img_res_path, resize_ratio, compos = detector.detect(input_screenshot, debug=True)
        
        duration = time.time() - start_time
        logger.info(f"Detection completed in {duration:.2f} seconds.")
        logger.info(f"Detected {len(compos)} components.")
        logger.info(f"Result image saved at: {img_res_path}")
        logger.info(f"Resize ratio: {resize_ratio}")

        # 5. Verify Outputs
        # Check subdirectories
        ocr_dir = os.path.join(output_dir, "ocr")
        ip_dir = os.path.join(output_dir, "ip")
        merge_dir = os.path.join(output_dir, "merge")

        print("\n--- Output Verification ---")
        for d in [ocr_dir, ip_dir, merge_dir]:
            if os.path.exists(d):
                files = os.listdir(d)
                print(f"Directory {d}: {len(files)} files found.")
            else:
                print(f"Directory {d}: NOT FOUND (This might indicate a failure in that stage)")

        # 6. Sample of detected text
        print("\n--- Detection Samples (First 5) ---")
        for i, compo in enumerate(compos[:5]):
            print(f"[{i}] Type: {compo.get('class', 'N/A')}, Text: {compo.get('text_content', 'N/A')}, Bbox: {compo.get('column_min')}, {compo.get('row_min')}, {compo.get('column_max')}, {compo.get('row_max')}")

    except Exception as e:
        logger.exception(f"An error occurred during detection: {e}")

if __name__ == "__main__":
    test_uied()
