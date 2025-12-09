![banner](assets/mast3r.jpg)

# MASt3R - Point Tracking Experiments

Official implementation of `Grounding Image Matching in 3D with MASt3R`
[[Project page](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)], [[MASt3R arxiv](https://arxiv.org/abs/2406.09756)], [[DUSt3R arxiv](https://arxiv.org/abs/2312.14132)]

This repository includes interactive point tracking tools for tracking points across multiple images using MASt3R.

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.

```python
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
```

## Installation

1. Clone MASt3R.
```bash
git clone --recursive https://github.com/unleashlive/mast3r
cd mast3r
# if you have already cloned mast3r:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n mast3r python=3.11 cmake=3.14.0
conda activate mast3r
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
pip install -r dust3r/requirements_optional.txt
```

3. Install additional dependencies for point tracking experiments.
```bash
# Python package
pip install roma

# System package for image viewer (Arch)
sudo pacman -S viewnior
# Or for Ubuntu/Debian: sudo apt install viewnior
```

4. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

## Quick Start - Point Tracking

### Prepare Your Images

Place your images in the `samples/` directory:
```bash
mkdir -p samples
# Copy your images to samples/
# e.g., cp /path/to/images/*.jpg samples/
```
The reference image picked will be the first from the list sorted alphabetically.

### 1. Interactive Point Tracking

Click on a reference image to track points across all other images interactively.

```bash
# Basic usage (opens interactive window)
python visualize_point_tracking.py

# With height filtering for elevated features (e.g., power lines)
python visualize_point_tracking.py --min-height 1.5

# Custom height tolerance
python visualize_point_tracking.py --height-tolerance 0.3
```

**How it works:**
1. A window opens showing the first image (reference)
2. Click anywhere on the image to track that point
3. Processing starts automatically in the background
4. Results open in viewnior image viewer showing matches across all images
5. Click another point to track it
6. Press 'q' to exit

**Features:**
- Interactive clicking interface
- Automatic processing in background
- Results displayed in 2-column layout (2x larger)
- Opens automatically in viewnior
- Height-based filtering for better matches
- Results saved to `samples/matches/`

**Options:**
- `--device cuda|cpu` - Device for inference (default: cuda)
- `--image_size SIZE` - Inference resolution (default: 512)
- `--samples_dir PATH` - Directory with images (default: ./samples)
- `--min-height VALUE` - Absolute height threshold for filtering
- `--height-tolerance VALUE` - Relative height tolerance (default: 0.5)

### 2. Global Point Tracking (Command Line)

Track a single point from command line without interactive GUI.

```bash
# Track a specific point (x, y coordinates)
python visualize_point_tracking_global.py --point 861,1215 --no-display

# With height filtering
python visualize_point_tracking_global.py --point 1500,2000 --min-height 1.5 --no-display
```

**Options:**
- `--point X,Y` - Point coordinates to track (required)
- `--no-display` - Don't show interactive display, only save files
- `--device cuda|cpu` - Device for inference (default: cuda)
- `--image_size SIZE` - Inference resolution (default: 512)
- `--samples_dir PATH` - Directory with images (default: ./samples)
- `--min-height VALUE` - Absolute height threshold for filtering
- `--height-tolerance VALUE` - Relative height tolerance (default: 0.5)

### Output

All visualizations are saved to `samples/matches/` with timestamped filenames.

Each visualization shows:
- Reference image paired with each target image side-by-side
- Red line connecting the query point to its match
- Red star marking the query point
- Coordinates and distance metrics

## Requirements

- Python 3.11+
- PyTorch with CUDA support
- viewnior image viewer (for interactive mode): `sudo pacman -S viewnior` (Arch/Manjaro)
- Tkinter for GUI (interactive mode only)

## Troubleshooting

### Tkinter Issues

If you see Tkinter errors:
```bash
# Upgrade uv (if using uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reinstall Python with Tkinter support
uv python upgrade --reinstall

# Recreate virtual environment
./update_venv.sh
```

### Missing Dependencies

```bash
# If roma is missing
pip install roma

# If viewnior is missing
sudo pacman -S viewnior  # Arch/Manjaro
sudo apt install viewnior  # Ubuntu/Debian
```

### Model Not Found

The model will be downloaded automatically on first run. If you have issues:
```bash
# Download manually
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

## More Information

For the original MASt3R demo and training code, see the [full documentation](https://github.com/naver/mast3r).

## Citation

```bibtex
@misc{mast3r_eccv24,
      title={Grounding Image Matching in 3D with MASt3R},
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      booktitle = {ECCV},
      year = {2024}
}
```
