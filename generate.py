from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import CLIPVisionModel

# image_encoder = CLIPVisionModel.from_pretrained("openai/")
# pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", image_encoder=image_encoder)

pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
pipe.load_lora_weights("Remade-AI/matrix-shot")

prompt = "b4ll3t t1m3 bullet time shot. The video captures a basketball player soaring mid-air in the midst of a powerful dunk, just inches away from the hoop. His right arm is cocked back, gripping the ball with precision, while his body twists dynamically with explosive energy. The shot is captured using a b4ll3t t1m3 bullet time shot effect, freezing the motion at the peak of the jump—highlighting the taut muscles, flared jersey, and shadow play against a clear blue sky. As the player hovers, the camera smoothly arcs around him from behind and to the left, circling from a rear angle to a dramatic side profile that emphasizes the vertical leap and the imminent slam. The basketball net ripples in anticipation, and even tiny particles of dust stirred by the jump hang suspended in the air, showcasing the intensity of this iconic dunk moment in slow-motion glory."
input_image = load_image("./images/01.jpeg")

frames = pipe(image=input_image, prompt=prompt).frames
export_to_video(frames, "./output/output.mp4")