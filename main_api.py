import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import shutil
import uuid
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Attempt to import rembg
try:
    from rembg import remove as rembg_remove
except ImportError:
    print("Rembg library not found. Functionality will be limited. Please install it: pip install rembg")
    print("You might also need to install specific ONNX runtime versions if prompted.")
    rembg_remove = None

# --- Configuration ---
STATIC_FILES_DIR = "static_glb_files"
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(STATIC_FILES_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Model Variables ---
# These will be loaded at startup
midas_model_global = None
midas_transform_global = None
midas_device_global = None

# --- FastAPI App Initialization ---
app = FastAPI(title="2D to 3D MVP API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity in MVP. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods.
    allow_headers=["*"], # Allows all headers.
)

# --- Mount Static Files Directory ---
# This makes files in STATIC_FILES_DIR accessible via /static_glb_files URL path
app.mount(f"/{STATIC_FILES_DIR}", StaticFiles(directory=STATIC_FILES_DIR), name="static_glb_files")

# --- Simple Root Endpoint --- 
@app.get("/", response_class=HTMLResponse)
async def read_root_html():
    html_content = """
    <html>
        <head>
            <title>2D to 3D MVP API</title>
        </head>
        <body>
            <h1>Welcome to the 2D to 3D Model Conversion API!</h1>
            <p>This API converts 2D images to 3D GLB models.</p>
            <p>To use the API, send a POST request with an image file to the 
               <code>/api/v1/convert_to_3d</code> endpoint.
            </p>
            <p>For detailed API documentation and to try it out, please visit 
               <a href="/docs">/docs</a> (Swagger UI) or 
               <a href="/redoc">/redoc</a> (ReDoc).
            </p>
            <p>Generated GLB files will be available under the 
               <code>/static_glb_files/</code> path once created.
            </p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- Model Loading (Adapted from mvp_backend_processor.py) ---
def load_midas_model_globally(model_type="MiDaS_small"):
    global midas_model_global, midas_transform_global, midas_device_global
    logger.info(f"Attempting to load MiDaS model: {model_type} from local path")
    # Define the local path to the MiDaS repository
    local_midas_repo_path = r'C:\Users\Hanah\.cache\torch\hub\MiDaS-master'
    try:
        # Load model from local path
        model = torch.hub.load(local_midas_repo_path, model_type, source='local', trust_repo=True)
        transform = None
        if model_type in ["MiDaS_small", "midas_v21_small"]:
            # Load transforms from local path
            transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).small_transform
        elif "dpt_beit" in model_type.lower() or \
             "dpt_swin" in model_type.lower() or \
             model_type in ["dpt_large", "dpt_hybrid", "dpt_hybrid_midas"]:
            if model_type == "dpt_beit_large_512":
                transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_beit_large_512_transform
            elif model_type == "dpt_beit_base_384":
                transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_beit_base_384_transform
            elif model_type == "dpt_swin2_large_384":
                transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_swin2_large_384_transform
            elif model_type == "dpt_swin2_base_384":
                transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_swin2_base_384_transform
            elif model_type == "dpt_swin2_tiny_256":
                transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_swin2_tiny_256_transform
            elif model_type == "dpt_large_midas": 
                 transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).dpt_large_midas_transform
            else: 
                logger.info(f"Using general 'dpt_transform' for model: {model_type}")
                transform = torch.hub.load(local_midas_repo_path, "dpt_transform", source='local', trust_repo=True)
        elif "midas_v21" in model_type and "small" not in model_type:
            transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).midas_v21_transform
        else:
            logger.warning(f"Using a default MiDaS transform ('small_transform') for model type: {model_type}.")
            transform = torch.hub.load(local_midas_repo_path, "transforms", source='local', trust_repo=True).small_transform
        
        if transform is None: raise RuntimeError(f"Could not determine transform for {model_type}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        midas_model_global = model
        midas_transform_global = transform
        midas_device_global = device
        logger.info(f"MiDaS model '{model_type}' loaded on {device} with its transform.")
    except Exception as e:
        logger.error(f"Error loading MiDaS model or transform for '{model_type}': {e}", exc_info=True)
        # Exit if essential models can't load, or handle gracefully
        raise RuntimeError(f"MiDaS model loading failed: {e}")

# --- Core Processing Functions (Adapted from mvp_backend_processor.py) ---
def segment_foreground_api(image_rgb_np):
    if rembg_remove is None:
        logger.warning("Rembg is not available. Skipping foreground segmentation.")
        image_rgba_pil = Image.fromarray(image_rgb_np).convert("RGBA")
        mask_np = np.ones((image_rgb_np.shape[0], image_rgb_np.shape[1]), dtype=np.uint8) * 255
        return image_rgba_pil, mask_np
    logger.info("Segmenting foreground with rembg...")
    try:
        image_pil = Image.fromarray(image_rgb_np)
        img_byte_arr = BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        result_bytes = rembg_remove(img_byte_arr)
        image_rgba_foreground_pil = Image.open(BytesIO(result_bytes)).convert("RGBA")
        alpha_channel = np.array(image_rgba_foreground_pil)[:, :, 3]
        foreground_mask_np = np.where(alpha_channel > 128, 255, 0).astype(np.uint8)
        logger.info("Foreground segmented.")
        return image_rgba_foreground_pil, foreground_mask_np
    except Exception as e:
        logger.error(f"Error during foreground segmentation: {e}", exc_info=True)
        image_rgba_pil = Image.fromarray(image_rgb_np).convert("RGBA")
        mask_np = np.ones((image_rgb_np.shape[0], image_rgb_np.shape[1]), dtype=np.uint8) * 255
        return image_rgba_pil, mask_np

def preprocess_image_api(image_bytes, transform_func):
    logger.info("Preprocessing image for MiDaS...")
    try:
        img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_rgb_np = np.array(img_pil)
        input_batch = transform_func(img_rgb_np)
        return input_batch, img_rgb_np
    except Exception as e:
        logger.error(f"Error in preprocess_image_api: {e}", exc_info=True)
        return None, None

def get_depth_map_api(model, input_batch, device, original_height, original_width):
    logger.info("Generating depth map...")
    try:
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
        logger.info("Depth map generated.")
        return depth_map_np
    except Exception as e:
        logger.error(f"Error in get_depth_map_api: {e}", exc_info=True)
        return None

def create_glb_from_depth_and_texture_api(depth_map_np, texture_image_rgba_pil, foreground_mask_np, output_glb_path, depth_scale_factor=0.1, thickness=0.05):
    logger.info(f"Creating solid GLB: {output_glb_path}")
    try:
        # --- DEBUG: Save the texture_image_rgba_pil to a file for inspection ---
        debug_texture_dir = "debug_textures"
        os.makedirs(debug_texture_dir, exist_ok=True)
        debug_texture_path = os.path.join(debug_texture_dir, f"texture_{uuid.uuid4()}.png")
        try:
            texture_image_rgba_pil.save(debug_texture_path, "PNG")
            logger.info(f"[DEBUG] Saved RGBA texture to: {debug_texture_path}")
        except Exception as e_save:
            logger.error(f"[DEBUG] Failed to save RGBA texture: {e_save}")
        # --- END DEBUG ---

        import trimesh
        # trimesh_logging = logging.getLogger('trimesh') # Already configured globally if needed
        # trimesh_logging.setLevel(logging.WARNING)

        H, W = depth_map_np.shape
        min_depth_val, max_depth_val = np.min(depth_map_np), np.max(depth_map_np)
        normalized_depth_map = (depth_map_np - min_depth_val) / (max_depth_val - min_depth_val) if max_depth_val > min_depth_val else np.zeros_like(depth_map_np)
        
        front_vertices, front_uvs, vertex_indices_map = [], [], {}
        for r in range(H): 
            for c in range(W): 
                x, y = c / (W - 1) - 0.5, (H - 1 - r) / (H - 1) - 0.5
                z_offset = (normalized_depth_map[r, c] if foreground_mask_np[r, c] > 0 else 0) - 0.5
                z = z_offset * depth_scale_factor
                idx = len(front_vertices)
                front_vertices.append([x,y,z]); vertex_indices_map[(r,c)] = idx
                front_uvs.append([c/(W-1), 1.0-(r/(H-1))])
        
        front_vertices_np, front_uvs_np = np.array(front_vertices, dtype=np.float32), np.array(front_uvs, dtype=np.float32)
        front_faces = [[vertex_indices_map[(r,c)], vertex_indices_map[(r+1,c)], vertex_indices_map[(r,c+1)],
                        vertex_indices_map[(r,c+1)], vertex_indices_map[(r+1,c)], vertex_indices_map[(r+1,c+1)]]
                       for r in range(H-1) for c in range(W-1)]
        front_faces_np = np.array(front_faces, dtype=np.int32).reshape(-1,3)

        if not front_vertices_np.size or not front_faces_np.size: raise ValueError("No front geometry generated.")

        back_vertices_np = front_vertices_np.copy(); back_vertices_np[:, 2] -= thickness
        all_vertices = np.vstack((front_vertices_np, back_vertices_np))
        num_front_v = len(front_vertices_np)
        all_faces = list(front_faces_np)
        all_faces.extend([[f[0]+num_front_v, f[2]+num_front_v, f[1]+num_front_v] for f in front_faces_np]) # Back faces

        def create_side_quad_api(v0_f, v1_f, v0_b, v1_b, flip=False):
            return [[v0_f,v0_b,v1_b],[v0_f,v1_b,v1_f]] if flip else [[v0_f,v1_f,v1_b],[v0_f,v1_b,v0_b]]

        for c in range(W-1): # Top & Bottom
            all_faces.extend(create_side_quad_api(vertex_indices_map[(0,c)], vertex_indices_map[(0,c+1)], vertex_indices_map[(0,c)]+num_front_v, vertex_indices_map[(0,c+1)]+num_front_v))
            all_faces.extend(create_side_quad_api(vertex_indices_map[(H-1,c)], vertex_indices_map[(H-1,c+1)], vertex_indices_map[(H-1,c)]+num_front_v, vertex_indices_map[(H-1,c+1)]+num_front_v,flip=True))
        for r in range(H-1): # Left & Right
            all_faces.extend(create_side_quad_api(vertex_indices_map[(r,0)], vertex_indices_map[(r+1,0)], vertex_indices_map[(r,0)]+num_front_v, vertex_indices_map[(r+1,0)]+num_front_v,flip=True))
            all_faces.extend(create_side_quad_api(vertex_indices_map[(r,W-1)], vertex_indices_map[(r+1,W-1)], vertex_indices_map[(r,W-1)]+num_front_v, vertex_indices_map[(r+1,W-1)]+num_front_v))

        solid_mesh = trimesh.Trimesh(vertices=all_vertices, faces=np.array(all_faces, dtype=np.int32), process=True)
        solid_mesh.fix_normals(multibody=False)
        all_uvs = np.vstack((front_uvs_np, front_uvs_np)) # Simplistic UVs for back/sides
        material = trimesh.visual.texture.SimpleMaterial(image=texture_image_rgba_pil)
        solid_mesh.visual = trimesh.visual.TextureVisuals(uv=all_uvs, material=material)
        
        glb_data = solid_mesh.export(file_type='glb')
        with open(output_glb_path, 'wb') as f: f.write(glb_data)
        logger.info(f"Solid GLB file saved to {output_glb_path}")
        return True
    except ImportError:
        logger.error("Trimesh library not installed.", exc_info=True)
    except Exception as e:
        logger.error(f"Error in create_glb_from_depth_and_texture_api: {e}", exc_info=True)
    return False

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading MiDaS model...")
    # Load a default model type, can be configured via env var later
    # For MVP, hardcoding 'dpt_hybrid_midas' or 'MiDaS_small'
    # Using 'MiDaS_small' for faster startup in MVP
    load_midas_model_globally(model_type="MiDaS_small") 
    if midas_model_global is None:
        logger.error("FATAL: MiDaS model could not be loaded. API may not function correctly.")
    logger.info("MiDaS Model loading process complete.")
    if rembg_remove is None:
        logger.warning("Rembg not loaded. Background removal will be skipped.")


# --- API Endpoint ---
@app.post("/api/v1/convert_to_3d")
async def convert_image_to_3d_endpoint(
    file: UploadFile = File(...),
    # Add other parameters like depth_scale, thickness if you want to control them via API
    # For MVP, we can use defaults or hardcode them in the call below
    depth_scale: float = 0.1,
    thickness: float = 0.03,
    model_type_override: str = None # Allow overriding global model for this call (optional)
):
    if midas_model_global is None or midas_transform_global is None:
        raise HTTPException(status_code=503, detail="MiDaS model is not available. API is not ready.")

    # Determine MiDaS model to use for this request
    current_midas_model = midas_model_global
    current_midas_transform = midas_transform_global
    current_midas_device = midas_device_global

    # Temporary logic if we allow model override per request (adds complexity for loading)
    # For MVP, it's simpler to rely on the globally loaded model.
    # if model_type_override and model_type_override != midas_model_global.name_or_path: # pseudo-code
    #     logger.info(f"Request to use model: {model_type_override}, not yet implemented for per-request loading")
    #     # Potentially load model_type_override here if not already loaded and cache it.
    #     # This requires more complex model management.

    try:
        # Save uploaded file temporarily
        temp_image_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded image saved to: {temp_image_path}")

        # Read image bytes for preprocessing (some transforms might want bytes directly)
        # Or use the saved path if functions are adapted for paths
        with open(temp_image_path, "rb") as f_bytes:
            image_bytes_content = f_bytes.read()

        # 1. Preprocess image
        input_batch, original_rgb_np = preprocess_image_api(image_bytes_content, current_midas_transform)
        if input_batch is None or original_rgb_np is None:
            raise HTTPException(status_code=500, detail="Image preprocessing failed.")
        
        original_height, original_width = original_rgb_np.shape[:2]

        # 2. Segment foreground
        segmented_rgba_pil, foreground_mask_np = segment_foreground_api(original_rgb_np)

        # 3. Get depth map
        depth_map_np = get_depth_map_api(current_midas_model, input_batch, current_midas_device, original_height, original_width)
        if depth_map_np is None:
            raise HTTPException(status_code=500, detail="Depth map generation failed.")

        # 4. Create GLB
        output_filename_glb = f"{uuid.uuid4()}.glb"
        output_glb_server_path = os.path.join(STATIC_FILES_DIR, output_filename_glb)
        
        success = create_glb_from_depth_and_texture_api(
            depth_map_np, 
            segmented_rgba_pil, 
            foreground_mask_np, 
            output_glb_server_path, 
            depth_scale_factor=depth_scale,
            thickness=thickness
        )

        if not success:
            raise HTTPException(status_code=500, detail="GLB file creation failed.")

        # Construct URL for the frontend to access the GLB file
        # Assuming API is running on http://localhost:8000 (adjust if different)
        # The URL path should match how StaticFiles is mounted
        glb_url = f"/{STATIC_FILES_DIR}/{output_filename_glb}" 
        
        return {"model_url": glb_url, "message": "GLB model generated successfully."}

    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error in convert_to_3d_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Clean up temporary uploaded file
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Cleaned up temporary file: {temp_image_path}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up temp file {temp_image_path}: {e_clean}")

# --- Main block to run Uvicorn (for direct execution, e.g. python main_api.py) ---
if __name__ == "__main__":
    import uvicorn
    # Default model for startup, can be overridden by environment variable or config
    startup_model_type = os.getenv("MIDAS_MODEL_TYPE", "MiDaS_small") 
    # Manually call model loading here if not relying on @app.on_event for direct run context
    # However, for uvicorn a better practice is to let @app.on_event handle it.
    
    # Note: Uvicorn should be started from the command line for production:
    # uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
    # The @app.on_event("startup") will handle model loading when uvicorn starts the app.
    
    # If you run `python main_api.py`, the startup event might not fire as expected
    # compared to `uvicorn main_api:app`. For direct run testing of model load:
    if not (midas_model_global and midas_transform_global): # Check if startup event already ran
         logger.info(f"Running __main__: Attempting to load MiDaS model '{startup_model_type}' for direct script execution context.")
         try:
            load_midas_model_globally(model_type=startup_model_type)
         except RuntimeError as e:
             logger.error(f"Failed to load model in __main__: {e}")
             # Decide if to exit or continue with a non-functional API for other routes.

    uvicorn.run(app, host="0.0.0.0", port=8000) 