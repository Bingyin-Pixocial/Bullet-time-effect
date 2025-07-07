# Bullet Time LoRA

A collection of LoRA (Low-Rank Adaptation) models and scripts for generating bullet time effects in videos using AI image-to-video models. This project provides implementations for both LTX-Video and Wan2.1-I2V models with specialized bullet time LoRA weights.

## 🎯 Features

- **Multiple Model Support**: Works with both LTX-Video and Wan2.1-I2V models
- **Bullet Time LoRA**: Pre-trained LoRA weights for authentic bullet time effects
- **Flexible Generation**: Support for various scenes including sports, action, and cinematic shots
- **High-Quality Output**: Optimized parameters for smooth, high-fidelity video generation
- **Easy-to-Use Scripts**: Ready-to-run Python scripts with example prompts

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd bullet_time_lora
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have CUDA-compatible GPU with sufficient VRAM (recommended: 16GB+)

### Basic Usage

#### Using LTX-Video Model

```python
python ltxv_generate.py
```

This script uses the LTX-Video model with bullet time LoRA weights to generate videos from input images.

#### Using Wan2.1-I2V Model

```python
python wan_generate.py
```

This script uses the Wan2.1-I2V model with matrix-shot LoRA for bullet time effects.

## 📁 Project Structure

```
bullet_time_lora/
├── ltxv_generate.py      # LTX-Video generation script
├── wan_generate.py       # Wan2.1-I2V generation script
├── generate.py           # Basic generation example
├── Bullet Time Effect.ipynb  # Jupyter notebook with examples
├── requirements.txt      # Python dependencies
├── images/              # Input images for generation
│   ├── 01.jpeg         # Basketball player
│   ├── 02.jpg          # Bird
│   ├── woman.jpeg      # Woman on bridge
│   ├── car.jpeg        # Classic car
│   ├── stunt.jpeg      # Stunt scene
│   └── warrior.jpg     # Spartan warrior
├── output/              # Generated video outputs
└── models/              # Model storage (if saving locally)
```

## 🎬 Example Prompts

The project includes several pre-configured prompts for different scenarios:

### Basketball Player
```
"bullet-time, a basketball player"
```

### Bird in Flight
```
"bullet-time, a bird"
```

### Woman on Bridge
```
"bullet-time, a young woman posing on a bridge over a scenic river on a bright, sunny day"
```

### Classic Car
```
"bullet-time, a classic red convertible car, most likely a 1960s Ford Mustang, parked on a grassy field under a cloudy sky camera arc view, ultra-photorealistic, high-detail, no grain"
```

### Action Stunt
```
"bullet-time, a dramatic explosion scene, likely from an action movie or a stunt performance"
```

## ⚙️ Configuration

### LTX-Video Parameters
- `num_frames`: 120 (default)
- `num_inference_steps`: 100
- `guidance_scale`: 6.5
- `guidance_rescale`: 0.75
- `decode_timestep`: 0.025
- `decode_noise_scale`: 0.012
- `fps`: 16

### Wan2.1-I2V Parameters
- `num_inference_steps`: 50
- `guidance_scale`: 7.5
- `lora_scale`: 1.0

## 🎨 Customization

### Adding New Images
1. Place your input image in the `images/` directory
2. Update the image path in the generation script
3. Modify the prompt to match your scene

### Adjusting Parameters
- Increase `num_inference_steps` for higher quality (slower generation)
- Adjust `guidance_scale` to control prompt adherence
- Modify `num_frames` to change video length

## 🔧 Technical Details

### Models Used
- **LTX-Video-0.9.7-dev**: Lightricks' video generation model
- **Wan2.1-I2V-14B-480P**: Wan-AI's image-to-video model
- **Bullet Time LoRA**: Specialized weights for bullet time effects

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: 16GB+ recommended for optimal performance
- **RAM**: 32GB+ system RAM recommended

## 📝 Notes

- The scripts are configured for specific CUDA devices (cuda:1, cuda:4). Adjust the device ID in the scripts based on your setup.
- Generated videos are saved in MP4 format in the `output/` directory.
- The negative prompt is optimized to avoid common artifacts and quality issues.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is for research and educational purposes. Please ensure you comply with the licenses of the underlying models and datasets used.

## 🙏 Acknowledgments

- Lightricks for LTX-Video model
- Wan-AI for Wan2.1-I2V model
- Remade-AI for matrix-shot LoRA weights
- The open-source AI community for continuous innovation 