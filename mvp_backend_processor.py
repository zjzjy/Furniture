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
        if "dpt_hybrid_midas" in model_type.lower(): # Specific case for dpt_hybrid_midas
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_hybrid_midas_transform
        elif "dpt_beit_large_512" in model_type.lower(): # Specific case
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_beit_large_512_transform
        elif "dpt_beit_base_384" in model_type.lower(): # Specific case
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_beit_base_384_transform
        elif "dpt_swin2_large_384" in model_type.lower(): # Specific case
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_swin2_large_384_transform
        elif "dpt_swin2_base_384" in model_type.lower(): # Specific case
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_swin2_base_384_transform
        elif "dpt_swin2_tiny_256" in model_type.lower(): # Specific case
             transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_swin2_tiny_256_transform
        elif "dpt_large_midas" in model_type.lower(): # Older DPT Large
            transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_large_midas_transform
        elif "dpt_large" in model_type.lower() and "midas" not in model_type.lower() : # General DPT, often just "dpt_transform"
             transform = torch.hub.load("intel-isl/MiDaS", "dpt_transform")
        elif "beit" in model_type.lower(): # General BEiT
            transform = torch.hub.load("intel-isl/MiDaS", "beit_transform")
        elif "swin" in model_type.lower(): # General Swin
            transform = torch.hub.load("intel-isl/MiDaS", "swin_transform")
        elif "midas_v21_small" in model_type.lower(): # MiDaS v2.1 Small
            transform = torch.hub.load("intel-isl/MiDaS", "transforms").midas_v21_small_transform
        else: # For MiDaS_small (v2.0) or other fallbacks
            print(f"Using default small_transform for model type: {model_type}")
            transform = torch.hub.load("intel-isl/MiDaS", transform_type).small_transform


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        print(f"MiDaS model loaded on {device}")
        return model, transform, device
    except Exception as e:
        print(f"Error loading MiDaS model or its transform: {e}")
        print("Please ensure you have an internet connection for torch.hub.load,")
        print("the model type is correct, and any necessary dependencies for the model are met.")
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


def preprocess_image(image_path, transform_func):
    """
    Loads an image using OpenCV, converts to RGB, and applies MiDaS transforms.
    """
    print(f"Preprocessing image for MiDaS: {image_path}")
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
    
    img_rgb_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    try:        
        img_pil_for_transform = Image.fromarray(img_rgb_np)
        input_batch = transform_func(img_pil_for_transform)

    except Exception as e_transform:
        print(f"Error applying MiDaS transform (tried PIL): {e_transform}. Trying with NumPy array directly.")
        try:
            input_batch = transform_func(img_rgb_np)
        except Exception as e_transform_numpy:
            print(f"Error applying MiDaS transform with NumPy as well: {e_transform_numpy}")
            return None, img_rgb_np 
    
    return input_batch, img_rgb_np

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

def create_glb_from_depth_and_texture(depth_map_np, texture_image_rgba_pil, foreground_mask_np, output_glb_path, depth_scale_factor=0.1, thickness=0.05):
    """
    Creates a GLB file with a more "solid" look by attempting to add sides and a back.
    The depth map is used for the front surface. Background depth is flattened.
    The texture_image_rgba_pil (with transparency) is applied as a texture to the front.
    Sides and back are untextured or have a default color.
    """
    print(f"Creating solid GLB: {output_glb_path} with depth_scale: {depth_scale_factor}, thickness: {thickness}")

    try:
        import trimesh 
        from trimesh import util as trimesh_util
        import logging
        log = logging.getLogger('trimesh')
        log.setLevel(logging.WARNING)


        H, W = depth_map_np.shape
        
        min_depth_val = np.min(depth_map_np)
        max_depth_val = np.max(depth_map_np)
        
        background_z_value_normalized = 0 

        if max_depth_val == min_depth_val: 
            normalized_depth_map = np.zeros_like(depth_map_np)
        else:
            normalized_depth_map = (depth_map_np - min_depth_val) / (max_depth_val - min_depth_val)
        
        print(f"Overall depth map normalized. Min: {np.min(normalized_depth_map):.3f}, Max: {np.max(normalized_depth_map):.3f}")

        front_vertices = []
        front_uvs = []
        vertex_indices_map = {} 

        for r in range(H): 
            for c in range(W): 
                x_coord = c / (W - 1) - 0.5  
                y_coord = (H - 1 - r) / (H - 1) - 0.5 
                
                is_foreground = foreground_mask_np[r, c] > 0 

                if is_foreground:
                    z_coord_offset = normalized_depth_map[r, c] - 0.5 
                    z_coord = z_coord_offset * depth_scale_factor
                else:
                    z_coord_offset = background_z_value_normalized - 0.5
                    z_coord = z_coord_offset * depth_scale_factor
                                     
                current_vertex_idx = len(front_vertices)
                front_vertices.append([x_coord, y_coord, z_coord])
                vertex_indices_map[(r,c)] = current_vertex_idx
                
                u_coord = c / (W - 1)
                v_coord = 1.0 - (r / (H - 1)) 
                front_uvs.append([u_coord, v_coord])
        
        front_vertices_np = np.array(front_vertices, dtype=np.float32)
        front_uvs_np = np.array(front_uvs, dtype=np.float32)

        front_faces = []
        for r in range(H - 1):
            for c in range(W - 1):
                idx00 = vertex_indices_map[(r,c)]        
                idx01 = vertex_indices_map[(r,c+1)]  
                idx10 = vertex_indices_map[(r+1,c)]  
                idx11 = vertex_indices_map[(r+1,c+1)]
                front_faces.append([idx00, idx10, idx01])
                front_faces.append([idx01, idx10, idx11])
        
        front_faces_np = np.array(front_faces, dtype=np.int32)

        if len(front_vertices_np) == 0 or len(front_faces_np) == 0:
            print("Error: No front vertices or faces generated. Cannot create mesh.")
            with open(output_glb_path, 'w') as f: f.write("Error - No front vertices or faces generated.")
            return
        
        back_vertices_np = front_vertices_np.copy()
        back_vertices_np[:, 2] -= thickness 

        num_front_vertices = len(front_vertices_np)
        all_vertices = np.vstack((front_vertices_np, back_vertices_np))
        
        all_faces = list(front_faces_np)
        
        back_faces_offset = num_front_vertices
        for face in front_faces_np:
            all_faces.append([face[0] + back_faces_offset, 
                              face[2] + back_faces_offset, 
                              face[1] + back_faces_offset])
        
        for c in range(W - 1): # Top edge
            idx0 = vertex_indices_map[(0,c)]
            idx1 = vertex_indices_map[(0,c+1)]
            all_faces.extend(create_side_quad(idx0, idx1, idx0 + num_front_vertices, idx1 + num_front_vertices))
        for c in range(W - 1): # Bottom edge
            idx0 = vertex_indices_map[(H-1,c)]
            idx1 = vertex_indices_map[(H-1,c+1)]
            all_faces.extend(create_side_quad(idx0, idx1, idx0 + num_front_vertices, idx1 + num_front_vertices, flip_quad=True))
        for r in range(H - 1): # Left edge
            idx0 = vertex_indices_map[(r,0)]
            idx1 = vertex_indices_map[(r+1,0)]
            all_faces.extend(create_side_quad(idx0, idx1, idx0 + num_front_vertices, idx1 + num_front_vertices, flip_quad=True))
        for r in range(H - 1): # Right edge
            idx0 = vertex_indices_map[(r,W-1)]
            idx1 = vertex_indices_map[(r+1,W-1)]
            all_faces.extend(create_side_quad(idx0, idx1, idx0 + num_front_vertices, idx1 + num_front_vertices))

        final_mesh_vertices = all_vertices
        final_mesh_faces = np.array(all_faces, dtype=np.int32)
        
        solid_mesh = trimesh.Trimesh(vertices=final_mesh_vertices, faces=final_mesh_faces, process=True)
        try:
            solid_mesh.fix_normals(multibody=False) # multibody=False if it's meant to be a single coherent object
        except Exception as e_fix_normals:
            print(f"Warning: trimesh.fix_normals failed: {e_fix_normals}. Mesh normals might be inconsistent.")

        all_uvs = np.vstack((front_uvs_np, front_uvs_np)) 

        material = trimesh.visual.texture.SimpleMaterial(image=texture_image_rgba_pil)
        solid_mesh.visual = trimesh.visual.TextureVisuals(uv=all_uvs, material=material)

        print(f"Exporting solid mesh with {len(solid_mesh.vertices)} vertices and {len(solid_mesh.faces)} faces.")
        glb_data = solid_mesh.export(file_type='glb')
        with open(output_glb_path, 'wb') as f:
            f.write(glb_data)
        print(f"Solid GLB file saved to {output_glb_path}")

    except ImportError: 
        print("Trimesh library is not installed. Cannot create GLB.")
        print("Please install it: pip install trimesh[easy]")
        with open(output_glb_path, 'w') as f: f.write("Error - Trimesh not installed.")
    except Exception as e:
        print(f"An error occurred during solid GLB creation: {e}")
        import traceback
        traceback.print_exc()
        with open(output_glb_path, 'w') as f: f.write(f"Error during GLB creation: {e}")

def create_side_quad(v0_front, v1_front, v0_back, v1_back, flip_quad=False):
    """ Helper to create two triangles for a side quad ensuring outward normals. """
    if flip_quad: # Typically for bottom and left edges when viewed from outside
        return [[v0_front, v0_back, v1_back], [v0_front, v1_back, v1_front]] 
    else: # Typically for top and right edges
        return [[v0_front, v1_front, v1_back], [v0_front, v1_back, v0_back]]


def main():
    parser = argparse.ArgumentParser(description="MVP Backend Processor: Convert 2D image to 3D GLB with background removal and solid extrusion.")
    parser.add_argument("input_image", type=str, help="Path to the input image file (e.g., my_chair.jpg).")
    parser.add_argument("output_glb", type=str, help="Path to save the output GLB file (e.g., my_chair.glb).")
    parser.add_argument("--model_type", type=str, default="MiDaS_small", 
                        help="MiDaS model type (e.g., 'MiDaS_small', 'dpt_hybrid_midas', 'dpt_beit_large_512'). Check MiDaS/torch.hub for options.")
    parser.add_argument("--depth_scale", type=float, default=0.1,
                        help="Factor to scale the depth effect (prominence of the object). Default: 0.1")
    parser.add_argument("--thickness", type=float, default=0.05,
                        help="Thickness for the 'solid' extrusion of the object. Default: 0.05")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        return

    midas_model, midas_transform_func, device = load_midas_model(args.model_type)
    if midas_model is None or midas_transform_func is None:
        print("Exiting due to model loading failure.")
        return

    input_batch_for_midas, original_rgb_np = preprocess_image(args.input_image, midas_transform_func)
    if input_batch_for_midas is None or original_rgb_np is None:
        print("Exiting due to image preprocessing failure for MiDaS.")
        return
    
    original_height, original_width = original_rgb_np.shape[:2]

    segmented_rgba_pil, foreground_mask_np = segment_foreground(original_rgb_np)

    depth_map_np = get_depth_map(midas_model, input_batch_for_midas, device, original_height, original_width)
    if depth_map_np is None:
        print("Exiting due to depth map generation failure.")
        return
        
    try:
        depth_map_visual_raw = (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np)) * 255
        cv2.imwrite("depth_map_output_raw.png", depth_map_visual_raw.astype(np.uint8))
        print("Raw depth map visualization saved to depth_map_output_raw.png")
    except Exception as e:
        print(f"Could not save raw depth map visualization: {e}")

    create_glb_from_depth_and_texture(
        depth_map_np, 
        segmented_rgba_pil, 
        foreground_mask_np, 
        args.output_glb, 
        depth_scale_factor=args.depth_scale,
        thickness=args.thickness
    )

    print(f"Processing complete. Check {args.output_glb}.")

if __name__ == "__main__":
    main() 