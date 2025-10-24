import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Image processing function
def process_image(image, model, device, resize_width=700, resize_height=700):
    if model is None:
        return None, None
    
    # Preprocess the image
    image_np = np.array(image)[..., ::-1] / 255
    
    # Create transform with configurable resize dimensions
    # Resize function expects (height, width) order, so we swap the parameters
    transform = Compose([
        Resize(resize_width, resize_height, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()

    # Ensure the depth map is in the correct shape
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions
    
    # Normalize depth map to 0-1 range
    pred_disp_normalized = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    
    # Generate 16-bit grayscale version
    depth_16bit = (pred_disp_normalized * 65535).astype(np.uint16)
    depth_image_gray = Image.fromarray(depth_16bit, mode='I;16')
    
    # Generate colorized version using Spectral_r colormap
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp_normalized[None, ..., None], 0, 1, cmap=cmap).squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_image_color = Image.fromarray(depth_colored_hwc)
    
    return depth_image_gray, depth_image_color

# Gradio interface function
def gradio_interface(image, model_size, use_original_resolution, resize_width, resize_height):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据用户选择的模型大小加载不同的配置
    model_kwargs = {
        "large": dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        ),
        "base": dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        "small": dict(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
        )
    }

    # 根据用户选择的模型大小加载对应的 checkpoint
    if model_size == "large":
        checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")
    elif model_size == "base":
        checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")
    elif model_size == "small":
        checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"small/model.safetensors", repo_type="model")
    else:
        raise ValueError(f"Unknown model size: {model_size}")
        
    # 加载模型
    if model_size == "large":
        model = DepthAnything(**model_kwargs[model_size]).to(device)
    else:
        model = DepthAnythingV2(**model_kwargs[model_size]).to(device)
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model = model.to(device)
    
    if model is None:
        return None, None
    
    # Determine resize dimensions based on user choice
    if use_original_resolution:
        # Use original image dimensions
        original_height, original_width = image.size[1], image.size[0]  # PIL Image size is (width, height)
        final_resize_width = original_width
        final_resize_height = original_height
    else:
        # Use user-specified dimensions
        final_resize_width = resize_width
        final_resize_height = resize_height
    
    # 处理图像并返回结果（both grayscale and colorized）
    depth_image_gray, depth_image_color = process_image(image, model, device, final_resize_width, final_resize_height)
    
    return depth_image_gray, depth_image_color

# Wrapper function to handle both outputs
def process_image_and_store(image, model_size, use_original_resolution, resize_width, resize_height, display_mode):
    """Process image, store both outputs, and return the selected format"""
    gray_output, color_output = gradio_interface(image, model_size, use_original_resolution, resize_width, resize_height)
    
    if display_mode == "16-bit Grayscale":
        return gray_output, gray_output, color_output
    else:
        return color_output, gray_output, color_output

def switch_display_mode(display_mode, gray_output, color_output):
    """Switch between outputs without reprocessing"""
    if display_mode == "16-bit Grayscale":
        return gray_output
    else:
        return color_output

# 创建 Gradio 界面
with gr.Blocks() as iface:
    gr.Markdown("# Depth Estimation Demo")
    gr.Markdown("Upload an image and configure the processing settings. Both 16-bit grayscale and colorized depth maps will be generated. You can switch between them using the dropdown below.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            model_size = gr.Dropdown(choices=["large", "base", "small"], label="Model Size", value="large")
            use_original_resolution = gr.Checkbox(label="Use Original Resolution", value=False, info="Keep original image dimensions")
            resize_width = gr.Number(label="Resize Width", value=700, minimum=224, maximum=2048, step=1, info="Width for resizing (ignored if using original resolution)")
            resize_height = gr.Number(label="Resize Height", value=700, minimum=224, maximum=2048, step=1, info="Height for resizing (ignored if using original resolution)")
            submit_btn = gr.Button("Generate Depth Map", variant="primary")
        
        with gr.Column():
            display_mode = gr.Dropdown(
                choices=["16-bit Grayscale", "Colorized (Spectral_r)"], 
                label="Display Mode", 
                value="16-bit Grayscale",
                info="Choose which depth map to display (both are generated)"
            )
            output_image = gr.Image(type="pil", format="png", label="Depth Map Output")
    
    # Hidden state to store both outputs
    gray_state = gr.State()
    color_state = gr.State()
    
    # Connect the submit button to process and store outputs
    submit_btn.click(
        fn=process_image_and_store,
        inputs=[input_image, model_size, use_original_resolution, resize_width, resize_height, display_mode],
        outputs=[output_image, gray_state, color_state]
    )
    
    # Allow switching display mode without reprocessing
    display_mode.change(
        fn=switch_display_mode,
        inputs=[display_mode, gray_state, color_state],
        outputs=output_image
    )

# 启动 Gradio 界面
iface.launch(share=True)
