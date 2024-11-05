import os
import torch
import yaml
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import glob
import time

from llm import generate_prompt

'''
Script to read images from a dataset in YOLO format and 
generate new images in another directory using Stable Diffusion model.
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./bag3-stablediff"
os.makedirs(output_dir, exist_ok=True)
root_dir = "yolo-v8/glitter_baseline/"
# Load dataset data.yaml file
data_yaml_path = root_dir + "data.yaml"

with open(data_yaml_path, 'r') as stream:
    data = yaml.safe_load(stream)

train_image_dir = root_dir + data['train']
val_image_dir = data.get('val', None)
class_names = data['names']

# Load images from dataset
image_paths = sorted(glob.glob(os.path.join(train_image_dir, "*.jpg")))

# Load Image-to-Image Translation model
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)
x = 0
exec_time = time.time()

# Image-to-Image Translation Execution
for i, img_path in enumerate(image_paths):
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    init_image = Image.open(img_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    
    with open(label_path, 'r') as f:
        label_data = f.readlines()
    if len(label_data) == 0:
        continue
    class_id = int(label_data[0].split()[0])
    if class_id == 0:
        x += 1
        class_name = class_names[class_id]

        # Remember: Change the prompt according to the class you are using.
        #prompt = f"plastic bag, underwater litter, litter, marine, plastic bag, polluted, discarded, organic waste"
        
        # Use LLM to generate correct keywords for the specific class
        prompt = generate_prompt(class_name)
        print(f"{prompt}, Class: {class_name}, Size: {init_image.size}")
        
        # Generate new image with stable diffusion
        images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
        
        # Save Image
        output_image_path = os.path.join(output_dir, f"translated_image_{x}.png")
        images[0].save(output_image_path)
        
    if x >= 200:  # Max 200 images
        break

print(f"Images saved in directory: {output_dir}.")
print("Execution time:",  time.time() - exec_time)