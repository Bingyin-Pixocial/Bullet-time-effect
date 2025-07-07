import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video
from torchvision import transforms
import numpy as np
from PIL import Image

# to_tensor = transforms.ToTensor()  # converts PIL Image to tensor and normalizes to [0,1]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


# 1) Load in fp16 and auto-shard across your GPUs
pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", torch_dtype=torch.float16).to("cuda:1")


# 2) Load your LoRA weights (they'll inherit fp16/device settings)
pipe.load_lora_weights("Remade-AI/matrix-shot")
pipe.fuse_lora(lora_scale=1.0)
# pipe.save_pretrained("./models/wan_bullet_time_lora")


# 3) Swap to a faster, 2nd‐order scheduler (often 2–3× faster than DDIM)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# prompt = "b4ll3t t1m3 bullet time shot. The video captures a basketball player soaring mid-air in the midst of a powerful dunk, just inches away from the hoop. His right arm is cocked back, gripping the ball with precision, while his body twists dynamically with explosive energy. The shot is captured using a b4ll3t t1m3 bullet time shot effect, freezing the motion at the peak of the jump—highlighting the taut muscles, flared jersey, and shadow play against a clear blue sky. As the player hovers, the camera smoothly arcs around him from behind and to the left, circling from a rear angle to a dramatic side profile that emphasizes the vertical leap and the imminent slam. The basketball net ripples in anticipation, and even tiny particles of dust stirred by the jump hang suspended in the air, showcasing the intensity of this iconic dunk moment in slow-motion glory."
# # 4) Move your input image to the same device
# input_image = load_image("./images/01.jpeg")  # PIL Image

prompt_bird = "b4ll3t t1m3 bullet time shot Two hoplite warriors, clad in gleaming bronze armor and scarlet crested helmets, are frozen mid-clash on a windswept plain. One warrior lunges with spear thrust high, his muscles coiled and cloak billowing, while the other braces behind a round shield, every tendon taut against the impending strike. The b4ll3t t1m3 bullet time shot effect arrests the dust kicked up by their boots and the faint glint of sun on metal—each dent in the shield and rivet in the cuirass rendered in crystal clarity. The background reveals a sprawling battlefield, distant banners flickering, and blurred figures locked in combat. The camera executes a seamless 360° glide—from a low-angle front view that accentuates the spear’s lethal arc, around to a sweeping side profile that captures the warriors’ fierce expressions, and finishing with an overhead pan that frames them against the vast sky—showcasing the raw intensity of this ancient duel in suspended slow-motion glory."
input_image_bird = load_image("./images/02.jpg")

prompt_warrior = "b4ll3t t1m3 bullet time shot A lone Spartan warrior, fully armored with a plumed helmet and scarlet cloak, is captured mid-lunge atop a jagged mountain ledge, his spear poised to strike and shield raised in defense. The scene utilizes a b4ll3t t1m3 bullet time shot effect, freezing the warrior’s taut muscles, the rippling folds of his cloak, and every flake of stone kicked up by his boots. The background reveals a vast, craggy landscape beneath a swirling sky, all rendered in a soft, painterly blur that heightens the sense of isolation and drama. The camera smoothly rotates around the Spartan—starting with a low-angle front view that emphasizes his upward drive, gliding to a side profile that highlights the curve of his form against the horizon, and sweeping overhead to frame him against the boundless heavens—showcasing the warrior’s heroic charge in suspended slow-motion glory."
input_image_warrior = load_image("./images/warrior.jpg")

prompt_stunt = "b4ll3t t1m3 bullet time shot A stunt performer clad in flame-retardant gear is hurled mid-air by a roaring explosion, debris and twin plumes of teal and amber smoke frozen around him. The scene employs a b4ll3t t1m3 bullet time shot effect, arresting every shard of concrete, every spark, and the taut lines of his protective suit in crystalline detail. In the background, an abandoned warehouse set dissolves into soft blur, heightening the raw power of the blast. The camera then glides in a flawless 360° sweep—from a low front view that underscores the upward thrust, around to a dramatic side profile revealing the performer’s outstretched limbs and the jagged edge of the shockwave, finishing with an overhead tilt that frames the entire suspended moment against the smoky sky—showcasing this explosive stunt in suspended slow-motion glory."
input_image_stunt = load_image("./images/stunt.jpeg")

prompt_women = "b4ll3t t1m3 bullet time shot A young woman in a brown leather jacket and mirrored sunglasses leans on an ornate riverside bridge, auburn hair drifting as punts glide below and foliage sways—each ripple, oar poised mid-stroke, and wisp of breeze frozen in crystalline detail. The camera arcs from a low, tight front view capturing her confident gaze, sweeps laterally to unveil the river’s mirror-like surface and softly blurred punts, spirals upward to an overhead vantage of the stone arch and lush canopy, then reverses in a smooth back-arc—showcasing this tranquil riverside moment from every angle in suspended slow-motion glory."
input_image_women = load_image("./images/woman.jpeg")

negative_prompt = (
    "low quality, blurry, distorted, artifacts, watermark, text, logo, signature, moving objects,"
    "oversaturated, overexposed, underexposed, poorly drawn, bad anatomy, extra limbs, "
    "missing fingers, cropped, out of frame, duplicate, morbid, mutilated, deformed, "
    "ugly, disfigured, unnatural colors, unrealistic, noisy, grainy, jpeg artifacts"
)

# 5) Infer with fewer steps (e.g. 20–30 instead of 50–100)
out = pipe(
    image=input_image_women,
    prompt=prompt_women,
    negative_prompt=negative_prompt,
    num_inference_steps=50,   # ← fewer steps = faster, but slightly lower fidelity
    guidance_scale=7.5,       # keep your usual guidance
)
frames = out.frames

# print("Type of out.frames:", type(frames))
# print("Length of frames:", len(frames))
# print("Type and shape of first frame:", type(frames[0]), getattr(frames[0], 'shape', None))

# print("frames.shape:", frames.shape)
if frames.ndim == 5 and frames.shape[0] == 1:
    frames = frames[0]
# print("After squeeze, frames.shape:", frames.shape)

# If frames is a numpy array of shape (num_frames, H, W, 3)
processed_frames = []
for i in range(frames.shape[0]):
    frame = frames[i]  # shape (480, 832, 3)
    # Ensure dtype is uint8
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(frame)
    img = img.convert("RGB")
    processed_frames.append(img)

export_to_video(processed_frames, "./output/wan_bullet_time_women.mp4")    