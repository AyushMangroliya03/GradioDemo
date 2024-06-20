import gradio as gr
import subprocess
import os
import uuid
from queue import Queue
from threading import Thread
from datetime import datetime
from PIL import Image
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

# Define a global queue for processing requests
request_queue = Queue()

def process_request(user_id, image_dir, user_name):
    # Run the finetuning command
    command = [
        "autotrain", "dreambooth", "--train",
        "--model", "stabilityai/stable-diffusion-xl-base-1.0",
        "--project-name", user_id,
        "--image-path", image_dir,
        "--prompt", f"A photo of {user_name} wearing casual clothes and smiling.",
        "--resolution", "1024",
        "--batch-size", "1",
        "--num-steps", "500",
        "--gradient-accumulation", "4",
        "--lr", "1e-4",
        "--mixed-precision", "fp16"
    ]
    subprocess.run(command)
    
    # Assume the .safetensors file is saved in the user directory with a known name
    safetensors_path = os.path.join(image_dir, "pytorch_lora_weights.safetensors")

    # Load the model with the finetuned weights
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(safetensors_path, weight_name="pytorch_lora_weights.safetensors")
    
    # Generate images with the model
    prompt = f"A portrait of {user_name} in a snowy place sitting beside a bonfire"
    result = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=3)
    images = result.images
    
    # Save the generated images
    saved_dir = os.path.join("saved", user_id)
    os.makedirs(saved_dir, exist_ok=True)
    
    saved_image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(saved_dir, f"image_{i}.png")
        img.save(img_path)
        saved_image_paths.append(img_path)
    
    return saved_image_paths

def worker():
    while True:
        # Get a request from the queue
        user_id, image_dir, user_name = request_queue.get()
        try:
            # Process the request
            saved_image_paths = process_request(user_id, image_dir, user_name)
            print(f"Generated images saved at: {saved_image_paths}")
        finally:
            # Mark the request as done
            request_queue.task_done()

# Start the worker thread
Thread(target=worker, daemon=True).start()

def handle_request(files, user_name):
    # Create a unique user_id based on the current time and a random UUID
    user_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4]}"
    # Remove underscores and other complex characters for simplicity
    user_id = user_id.replace("-", "").replace("_", "")
    
    # Create a directory for the user's images
    user_dir = os.path.join("media", user_id)
    os.makedirs(user_dir, exist_ok=True)

    # Save the uploaded images to the user directory
    for i, file_path in enumerate(files):
        img = Image.open(file_path)
        img_path = os.path.join(user_dir, f"image_{i}.png")
        img.save(img_path)

    # Add the request to the queue
    request_queue.put((user_id, user_dir, user_name))

    return f"Request submitted for user {user_name}. Your user ID is {user_id}. The images will be processed and saved in the directory 'saved/{user_id}'."

# Create a Gradio interface
iface = gr.Interface(
    fn=handle_request,
    inputs=[
        gr.Files(label="Upload Images", file_count="multiple", type="filepath"),
        gr.Textbox(label="User Name")
    ],
    outputs="text",
    title="Stable Diffusion Model Finetuning",
    description="Upload 5-10 images and provide your first name to finetune the Stable Diffusion model."
)

# Launch the Gradio interface
iface.launch()
