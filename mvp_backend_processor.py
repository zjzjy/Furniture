import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO # Needed for rembg

# Attempt to import rembg, provide instructions if missing
try:
    from rembg import remove
except ImportError:
    print("Rembg library not found. Please install it: pip install rembg")
    print("You might also need to install specific ONNX runtime versions if prompted, e.g.:")
    print("pip install onnxruntime-gpu  # For NVIDIA GPU")
    print("pip install onnxruntime      # For CPU")
    remove = None # Set to None if import fails

# Trimesh is crucial for mesh creation and GLB export.
# It will be imported conditionally within the function that uses it.

def load_midas_model(model_type="MiDaS_small", model_path=None):
    """
    Loads a MiDaS model.
    Tries to load from torch.hub, or a local path if provided in the reference repo.
    """
    print(f"Loading MiDaS model: {model_type}")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type) # Allow various model types
        transform_type = "transforms"
        # Adjust transform based on model type if necessary (some models have specific transforms)
        if "dpt" in model_type.lower() or "beit" in model_type.lower() or "swin" in model_type.lower() :
             # DPT, BEiT, Swin models typically use a different transform
            try:
                transform = torch.hub.load("intel-isl/MiDaS", "dpt_transform" if "dpt" in model_type.lower() else model_type.split('_')[0] + "_transform") # Heuristic
            except: # Fallback for older or differently named transforms
                 print(f"Specific transform for {model_type} not found by heuristic, using MiDaS default transform.")
                 transform = torch.hub.load("intel-isl/MiDaS", transform_type).default_transform
        else: # For older MiDaS_small etc.
            transform = torch.hub.load("intel-isl/MiDaS", transform_type).small_transform


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        print(f"MiDaS model loaded on {device}")
        return model, transform, device
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        print("Please ensure you have an internet connection for torch.hub.load,")
        print("or that models are locally available and correctly configured as per the reference repository.")
        return None, None, None

def segment_foreground(image_rgb_np):
    """
    Segments the foreground from an RGB image using rembg.
    Returns:
        - image_rgba_foreground_pil: PIL Image, RGBA with transparent background.
        - foreground_mask_np: Numpy array (H, W), 0 for background, 255 for foreground.
    """
    if remove is None:
        print("Rembg is not available. Skipping foreground segmentation.")
        # Return original image as RGBA and a full foreground mask
        image_rgba_pil = Image.fromarray(image_rgb_np).convert("RGBA")
        mask_np = np.ones((image_rgb_np.shape[0], image_rgb_np.shape[1]), dtype=np.uint8) * 255
        return image_rgba_pil, mask_np

    print("Segmenting foreground with rembg...")
    try:
        # Convert numpy RGB to PIL Image
        image_pil = Image.fromarray(image_rgb_np)
        
        # rembg expects image data as bytes
        img_byte_arr = BytesIO()
        image_pil.save(img_byte_arr, format='PNG') # Save to bytes buffer
        img_byte_arr = img_byte_arr.getvalue()

        # Use rembg to remove background
        result_bytes = remove(img_byte_arr)
        image_rgba_foreground_pil = Image.open(BytesIO(result_bytes)).convert("RGBA")

        # Create a binary mask from the alpha channel
        alpha_channel = np.array(image_rgba_foreground_pil)[:, :, 3]
        foreground_mask_np = np.where(alpha_channel > 128, 255, 0).astype(np.uint8) # Threshold alpha
        
        print("Foreground segmented.")
        # Save segmented image for inspection
        try:
            image_rgba_foreground_pil.save("segmented_foreground_output.png")
            Image.fromarray(foreground_mask_np).save("foreground_mask_output.png")
            print("Segmented foreground and mask saved to segmented_foreground_output.png and foreground_mask_output.png")
        except Exception as e_save:
            print(f"Could not save segmentation debug images: {e_save}")
            
        return image_rgba_foreground_pil, foreground_mask_np
    except Exception as e:
        print(f"Error during foreground segmentation with rembg: {e}")
        # Fallback: return original image as RGBA and a full foreground mask
        image_rgba_pil = Image.fromarray(image_rgb_np).convert("RGBA")
        mask_np = np.ones((image_rgb_np.shape[0], image_rgb_np.shape[1]), dtype=np.uint8) * 255
        return image_rgba_pil, mask_np


def preprocess_image(image_path, transform_func): # Renamed transform to transform_func
    """
    Loads an image using OpenCV, converts to RGB, and applies MiDaS transforms.
    """
    print(f"Preprocessing image for MiDaS: {image_path}")
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
    
    img_rgb_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # Apply the MiDaS-specific transform to the RGB numpy array
    # The transform function from torch.hub typically expects a PIL image or numpy array
    # and returns a tensor.
    try:
        # Some transforms expect PIL, some numpy. Let's try PIL first.
        img_pil_for_transform = Image.fromarray(img_rgb_np)
        input_batch = transform_func(img_pil_for_transform)
    except TypeError: # If transform expects numpy array directly or different format
        print("MiDaS transform failed with PIL, trying with NumPy array.")
        input_batch = transform_func(img_rgb_np) # This might need specific reshaping based on model
    except Exception as e_transform:
        print(f"Error applying MiDaS transform: {e_transform}")
        return None, img_rgb_np # Return original RGB for fallback or other uses

    return input_batch, img_rgb_np # Return original RGB for segmentation and MiDaS

def get_depth_map(model, input_batch, device, original_height, original_width):
    """
    Runs the MiDaS model to get a depth map and resizes it to original image dimensions.
    """
    print("Generating depth map...")
    with torch.no_grad():
        input_batch = input_batch.to(device)
        prediction = model(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map_np = prediction.cpu().numpy()
    print("Depth map generated.")
    return depth_map_np

def create_glb_from_depth_and_texture(depth_map_np, texture_image_rgba_pil, foreground_mask_np, output_glb_path, depth_scale_factor=0.1):
    """
    Creates a GLB file from a depth map and an RGBA texture image, using a foreground mask.
    The depth map is used to generate a height field mesh. Background depth is flattened.
    The texture_image_rgba_pil (with transparency) is applied as a texture.
    """
    print(f"Creating GLB: {output_glb_path} with depth_scale_factor: {depth_scale_factor}")

    try:
        import trimesh 
        from trimesh import util as trimesh_util

        H, W = depth_map_np.shape
        # texture_image_rgba_pil is already a PIL image

        min_depth = np.min(depth_map_np)
        max_depth = np.max(depth_map_np)
        
        # Base Z value for flat background areas
        # This ensures background parts of the mesh are co-planar.
        background_z_offset = 0 # Or use (min_depth - 0.5) if min_depth is not already a good base

        if max_depth == min_depth: 
            normalized_depth_map = np.zeros_like(depth_map_np)
        else:
            # Normalize entire depth map first
            normalized_depth_map = (depth_map_np - min_depth) / (max_depth - min_depth)
        
        print(f"Overall depth map normalized. Min: {np.min(normalized_depth_map):.3f}, Max: {np.max(normalized_depth_map):.3f}")

        vertices = []
        uvs = []
        for r in range(H): 
            for c in range(W): 
                x_coord = c / (W - 1) - 0.5  
                y_coord = (H - 1 - r) / (H - 1) - 0.5 
                
                is_foreground = foreground_mask_np[r, c] > 0 # Check mask (255 for fg)

                if is_foreground:
                    # Apply depth scaling only to foreground
                    # Subtract 0.5 to center the object's depth variation around z=0 before scaling
                    z_coord_offset = normalized_depth_map[r, c] - 0.5 
                    z_coord = z_coord_offset * depth_scale_factor
                else:
                    # Flatten background
                    # Adjust this if your desired background plane is different
                    z_coord = (background_z_offset - 0.5) * depth_scale_factor 
                                    
                vertices.append([x_coord, y_coord, z_coord])
                
                u_coord = c / (W - 1)
                v_coord = 1.0 - (r / (H - 1)) 
                uvs.append([u_coord, v_coord])

        vertices_np = np.array(vertices, dtype=np.float32)
        uvs_np = np.array(uvs, dtype=np.float32)

        faces = []
        for r in range(H - 1):
            for c in range(W - 1):
                idx00 = r * W + c        
                idx01 = r * W + (c + 1)  
                idx10 = (r + 1) * W + c  
                idx11 = (r + 1) * W + (c + 1)
                faces.append([idx00, idx10, idx01])
                faces.append([idx01, idx10, idx11])
        
        faces_np = np.array(faces, dtype=np.int32)

        if len(vertices_np) == 0 or len(faces_np) == 0:
            print("Error: No vertices or faces generated. Cannot create mesh.")
            with open(output_glb_path, 'w') as f:
                f.write("Error - No vertices or faces generated.")
            return

        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        
        # Ensure the material uses the RGBA PIL image for transparency
        material = trimesh.visual.texture.SimpleMaterial(image=texture_image_rgba_pil)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        
        print(f"Exporting mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        glb_data = mesh.export(file_type='glb')
        with open(output_glb_path, 'wb') as f:
            f.write(glb_data)
        print(f"GLB file saved to {output_glb_path}")

    except ImportError:
        print("Trimesh library is not installed. Cannot create GLB.")
        print("Please install it: pip install trimesh[easy]")
        with open(output_glb_path, 'w') as f:
            f.write("Error - Trimesh not installed. Mesh generation skipped.")
    except Exception as e:
        print(f"An error occurred during GLB creation: {e}")
        import traceback
        traceback.print_exc()
        with open(output_glb_path, 'w') as f:
            f.write(f"Error during GLB creation: {e}")


def main():
    parser = argparse.ArgumentParser(description="MVP Backend Processor: Convert 2D image to 3D GLB with background removal.")
    parser.add_argument("input_image", type=str, help="Path to the input image file (e.g., my_chair.jpg).")
    parser.add_argument("output_glb", type=str, help="Path to save the output GLB file (e.g., my_chair.glb).")
    parser.add_argument("--model_type", type=str, default="MiDaS_small", 
                        help="MiDaS model type (e.g., 'MiDaS_small', 'dpt_beit_large_512', 'dpt_hybrid_midas'). Check MiDaS/torch.hub for options.")
    parser.add_argument("--depth_scale", type=float, default=0.1,
                        help="Factor to scale the depth effect (thickness of the model). Default: 0.1")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        return

    # 1. Load MiDaS model
    midas_model, midas_transform_func, device = load_midas_model(args.model_type) # Renamed midas_transform to midas_transform_func
    if midas_model is None or midas_transform_func is None:
        print("Exiting due to model loading failure.")
        return

    # 2. Preprocess image for MiDaS (using original image for better depth context)
    #    and get the original RGB numpy array for segmentation.
    input_batch_for_midas, original_rgb_np = preprocess_image(args.input_image, midas_transform_func)
    if input_batch_for_midas is None or original_rgb_np is None:
        print("Exiting due to image preprocessing failure for MiDaS.")
        return
    
    original_height, original_width = original_rgb_np.shape[:2]

    # 3. Segment foreground from the original RGB image
    # This returns an RGBA PIL image (for texturing) and a numpy mask
    segmented_rgba_pil, foreground_mask_np = segment_foreground(original_rgb_np)
    # segmented_rgba_pil will be used as texture
    # foreground_mask_np will be used to flatten background depth

    # 4. Get depth map using MiDaS on the original image context
    depth_map_np = get_depth_map(midas_model, input_batch_for_midas, device, original_height, original_width)
    if depth_map_np is None:
        print("Exiting due to depth map generation failure.")
        return
        
    try:
        # Visualize the raw depth map from MiDaS
        depth_map_visual_raw = (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np)) * 255
        cv2.imwrite("depth_map_output_raw.png", depth_map_visual_raw.astype(np.uint8))
        print("Raw depth map visualization saved to depth_map_output_raw.png")
    except Exception as e:
        print(f"Could not save raw depth map visualization: {e}")

    # 5. Create GLB from depth map, segmented RGBA texture, and foreground mask
    create_glb_from_depth_and_texture(
        depth_map_np, 
        segmented_rgba_pil, # Use the RGBA image with transparent background as texture
        foreground_mask_np, # Use the mask to control depth of background
        args.output_glb, 
        depth_scale_factor=args.depth_scale
    )

    print(f"Processing complete. Check {args.output_glb}.")

if __name__ == "__main__":
    main() 