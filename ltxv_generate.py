import torch
import numpy as np
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image, export_to_video


def to_uint8_hwc(frame):
    # single‐frame helper
    if hasattr(frame, "convert"):
        arr = np.array(frame.convert("RGB"))
    else:
        arr = np.array(frame)
        if arr.ndim == 3 and arr.shape[0] in (1,3,4):
            arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0,255).astype(np.uint8)
    # collapse alpha or gray if needed
    if arr.ndim==3 and arr.shape[-1]==4:
        arr = arr[...,:3]
    if arr.ndim==2:
        arr = np.repeat(arr[:,:,None], 3, axis=-1)
    return arr

# — pipeline + LoRA loading —
pipeline = DiffusionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-dev",
    torch_dtype=torch.bfloat16
).to("cuda:4")


lora_path = hf_hub_download(
    "Lightricks/LTXV-LoRAs", "bullet_time_step_02250_comfy.safetensors"
)
pipeline.load_lora_weights(lora_path, adapter_name="bullet-time")
pipeline.set_adapters("bullet-time")


prompt = (
    "bullet-time,"
    "a basketball player"
)

prompt_bird = (
    "bullet-time,"
    "a bird"
)

prompt_woman = (
    "bullet-time,"
    "a young woman posing on a bridge over a scenic river on a bright, sunny day"
)

prompt_car = (
    "bullet-time,"
    "a classic red convertible car, most likely a 1960s Ford Mustang, parked on a grassy field under a cloudy sky"
    "camera arc view,"
    "ultra-photorealistic, high-detail, no grain"
)

prompt_stunt = (
    "bullet-time,"
    " a dramatic explosion scene, likely from an action movie or a stunt performance"
)


negative_prompt = (
    "low quality, blurry, distorted, artifacts, watermark, text, logo, signature, moving objects,"
    "oversaturated, overexposed, underexposed, poorly drawn, bad anatomy, extra limbs, "
    "missing fingers, cropped, out of frame, duplicate, morbid, mutilated, deformed, "
    "ugly, disfigured, unnatural colors, unrealistic, noisy, grainy, jpeg artifacts"
)
image = load_image("./images/01.jpeg")
image_bird = load_image("./images/02.jpg")
image_woman = load_image("./images/woman.jpeg")
image_car = load_image("./images/car.jpeg")
image_stunt = load_image("./images/stunt.jpeg")

# generate PIL frames
pil_frames = pipeline(
    prompt=prompt_woman,
    image=image_woman,
    num_frames=120,
    num_inference_steps=100,      # more steps = finer detail & fewer artifacts
    guidance_scale=6.5,           # moderate guidance to keep the prompt strong but not over-sharpened
    guidance_rescale=0.75,        # smooth out classifier-free guidance overshoot
    decode_timestep=0.025,        # smaller timestep for smoother interpolation
    decode_noise_scale=0.012,     # lower per-frame noise to reduce jitter
    negative_prompt=negative_prompt,
).frames

# Save the first frame for debugging
# If pil_frames[0] is a list, get the first image inside it
# first_img = pil_frames[0][0] if isinstance(pil_frames[0], list) else pil_frames[0]
# first_img.save("debug_first_frame.png")

# print("pil_frames[0]", pil_frames[0])
# convert to H×W×3 uint8 RGB
np_frames = [to_uint8_hwc(f) for f in pil_frames]
# print("np_frames[0].shape, np_frames[0].dtype", np_frames[0].shape, np_frames[0].dtype)  # should be (512, 768, 3) uint8
full = np_frames[0]
if isinstance(full, np.ndarray) and full.ndim == 4:
    frame_list = [ full[i] for i in range(full.shape[0]) ]
else:
    frame_list = np_frames

export_to_video(frame_list, "./output/ltxv_bullet_time_woman.mp4", fps=16)