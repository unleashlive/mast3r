#!/usr/bin/env python3
"""
Global Point Tracking Visualization for MASt3R

This script uses global alignment across all images to track points more accurately.
Instead of matching pairs independently, it leverages photogrammetric context from all views.

Usage:
    python visualize_point_tracking_global.py [--device cuda|cpu] [--image_size 512]
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image
import tempfile

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add MASt3R paths
import mast3r.utils.path_to_dust3r  # noqa

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.utils.schedules import cosine_schedule
from dust3r.utils.image import load_images
from dust3r.inference import inference
from mast3r.image_pairs import make_pairs


def select_point_interactively(image_path):
    """Display an image and allow user to click to select a point."""
    current_backend = matplotlib.get_backend()
    if current_backend == 'agg' or current_backend == 'Agg':
        for backend in ['QtAgg', 'Qt5Agg', 'TkAgg', 'GTK3Agg', 'WXAgg']:
            try:
                matplotlib.use(backend, force=True)
                import importlib
                importlib.reload(plt)
                print(f"Switched to {backend} backend for interactive display")
                break
            except:
                continue

    img = Image.open(image_path)
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array)
    ax.set_title('Click on a point to track across images\\n(Close window to use clicked point)')
    ax.axis('off')

    print("\\nPlease click on a point in the image...")
    print("Close the window when done.")

    points = plt.ginput(n=1, timeout=0)
    plt.close()

    if not points:
        print("No point selected!")
        return None

    x, y = points[0]
    print(f"Selected point: ({x:.1f}, {y:.1f})")

    return (float(x), float(y))


def visualize_single_match(ref_img_path, target_img_path, query_point, matched_point,
                          output_path, target_idx, show_display=True):
    """Create and save a visualization of a single match between two images."""
    # Load images
    img_ref = np.array(Image.open(ref_img_path))
    img_target = np.array(Image.open(target_img_path))

    # Pad images to same height
    H0, W0 = img_ref.shape[:2]
    H1, W1 = img_target.shape[:2]

    img_ref_pad = np.pad(img_ref, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                         'constant', constant_values=0)
    img_target_pad = np.pad(img_target, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                            'constant', constant_values=0)

    # Concatenate side-by-side
    img_combined = np.concatenate((img_ref_pad, img_target_pad), axis=1)

    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img_combined)

    # Draw query match in red
    x0, y0 = query_point
    x1, y1 = matched_point

    # Target image is offset by W0 horizontally
    ax.plot([x0, x1 + W0], [y0, y1], '-o',
            color='red', linewidth=3, markersize=12,
            alpha=1.0, scalex=False, scaley=False)

    # Add query point marker in reference image
    ax.plot(query_point[0], query_point[1], 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)

    ax.set_title(f'Point Tracking (Global Alignment): Reference → Image {target_idx}\\n' +
                f'Query Point: ({query_point[0]:.1f}, {query_point[1]:.1f}) → ' +
                f'Matched Point: ({matched_point[0]:.1f}, {matched_point[1]:.1f})',
                fontsize=14, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    if show_display:
        try:
            plt.show()
        except:
            pass
    plt.close()


def find_3d_point_from_2d(pts3d, intrinsics, cam2world, query_point_2d, min_height=None, search_radius=5):
    """
    Find the 3D point corresponding to a 2D image point.
    Searches in a neighborhood if the exact point is invalid.

    Args:
        pts3d: (H, W, 3) array of 3D points in world coordinates
        intrinsics: (3, 3) camera intrinsic matrix
        cam2world: (4, 4) camera-to-world transformation matrix
        query_point_2d: (x, y) tuple of 2D coordinates
        min_height: Optional minimum Z-coordinate for filtering
        search_radius: Radius to search for valid points if exact location fails

    Returns:
        3D point in world coordinates, or None if not found
    """
    x, y = int(round(query_point_2d[0])), int(round(query_point_2d[1]))

    H, W = pts3d.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return None

    # Try exact location first
    pt3d = pts3d[y, x]
    if np.isfinite(pt3d).all():
        if min_height is None or pt3d[2] >= min_height:
            return pt3d

    # Search in neighborhood
    best_pt = None
    best_dist = float('inf')

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nx, ny = x + dx, y + dy

            if not (0 <= nx < W and 0 <= ny < H):
                continue

            pt3d = pts3d[ny, nx]

            if not np.isfinite(pt3d).all():
                continue

            if min_height is not None and pt3d[2] < min_height:
                continue

            # Prefer points closer to the query location
            dist = np.sqrt(dx**2 + dy**2)
            if dist < best_dist:
                best_dist = dist
                best_pt = pt3d

    if best_pt is not None:
        print(f"  Found valid point at offset ({int(best_dist)} pixels)")

    return best_pt


def project_3d_to_2d(pt3d_world, intrinsics, cam2world):
    """
    Project a 3D world point to 2D image coordinates.

    Args:
        pt3d_world: (3,) array of 3D point in world coordinates
        intrinsics: (3, 3) camera intrinsic matrix
        cam2world: (4, 4) camera-to-world transformation matrix

    Returns:
        (x, y) tuple of 2D coordinates, or None if behind camera
    """
    # Transform from world to camera coordinates
    world2cam = np.linalg.inv(cam2world)
    pt3d_cam = world2cam[:3, :3] @ pt3d_world + world2cam[:3, 3]

    # Check if point is in front of camera
    if pt3d_cam[2] <= 0:
        return None

    # Project to image coordinates
    pt2d_homo = intrinsics @ pt3d_cam
    pt2d = pt2d_homo[:2] / pt2d_homo[2]

    return tuple(pt2d)


def main():
    parser = argparse.ArgumentParser(description='Track a point across multiple images using MASt3R global alignment')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for model inference')
    parser.add_argument('--samples_dir', type=str,
                       default='/home/mordka/UNLEASH/work/mast3r/samples',
                       help='Directory containing sample images')
    parser.add_argument('--point', type=str, default=None,
                       help='Query point coordinates as "x,y" (e.g., "1500,2000")')
    parser.add_argument('--ref-image', type=int, default=0,
                       help='Index of reference image (0-based, default: 0 = first image)')
    parser.add_argument('--min-height', type=float, default=None,
                       help='Minimum relative height (Z-coordinate) for filtering')
    parser.add_argument('--scenegraph', type=str, default='complete',
                       choices=['complete', 'swin', 'oneref'],
                       help='Scene graph type for image pairs')
    parser.add_argument('--winsize', type=int, default=1,
                       help='Window size for swin scene graph')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display visualizations interactively')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("MASt3R Global Point Tracking Visualization")
    print("=" * 80)

    # Find all images
    image_patterns = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(args.samples_dir, pattern)))

    image_files = sorted(image_files)

    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {args.samples_dir}")
        return 1

    print(f"\\nFound {len(image_files)} images:")
    for i, img_path in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(img_path)}")

    # Validate reference image index
    if args.ref_image < 0 or args.ref_image >= len(image_files):
        print(f"Error: Reference image index {args.ref_image} out of range [0, {len(image_files)-1}]")
        return 1

    ref_img_path = image_files[args.ref_image]
    print(f"\\nReference image: {os.path.basename(ref_img_path)} (index {args.ref_image})")

    # Get query point
    if args.point:
        try:
            x, y = map(float, args.point.split(','))
            query_point = (x, y)
            print(f"\\nUsing query point: ({x:.1f}, {y:.1f})")
        except ValueError:
            print(f"Error: Invalid point format '{args.point}'")
            return 1
    else:
        try:
            query_point = select_point_interactively(ref_img_path)
            if query_point is None:
                return 1
        except Exception as e:
            print(f"\\nError with interactive selection: {e}")
            return 1

    # Load MASt3R model
    print(f"\\nLoading MASt3R model on {args.device}...")
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(args.device)
    model.eval()
    print("Model loaded successfully!")

    # Load all images
    print(f"\\nLoading {len(image_files)} images for global alignment...")
    images = load_images(image_files, size=args.image_size)

    # Create image pairs for global alignment
    scene_graph = args.scenegraph
    if args.scenegraph == 'swin':
        scene_graph = f'swin-{args.winsize}'

    print(f"\\nCreating scene graph ({scene_graph})...")
    pairs = make_pairs(images, scene_graph=scene_graph,
                      prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")

    # Create persistent cache directory for this session
    cache_dir = os.path.join(args.samples_dir, '.cache_global')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\\nRunning global alignment...")
    print("This may take a few minutes...")

    scene = sparse_global_alignment(
        image_files, pairs, cache_dir, model,
        device=args.device,
        schedule=cosine_schedule, lr1=0.07, niter1=100,
        lr2=0.014, niter2=50,
        shared_intrinsics=False
    )

    print("\\nGlobal alignment complete!")

    # Get globally aligned 3D points and camera parameters
    pts3d_all = scene.get_sparse_pts3d()  # List of (N_i, 3) arrays per image
    intrinsics_all = [K.cpu().numpy() for K in scene.intrinsics]
    cam2world_all = [pose.cpu().numpy() for pose in scene.cam2w]

    # Get dense 3D points (now cache_dir still exists)
    pts3d_dense_all, depthmaps_all, confs_all = scene.get_dense_pts3d(clean_depth=True, subsample=8)

    print(f"\\nExtracted 3D reconstruction:")
    print(f"  Number of views: {len(pts3d_all)}")
    for i, pts in enumerate(pts3d_all):
        print(f"  View {i}: {len(pts)} 3D points")

    # Find 3D point in reference image
    print(f"\\nFinding 3D point for query location in reference image...")

    # Get original and inference resolutions
    ref_img_pil = Image.open(ref_img_path)
    ref_original_width, ref_original_height = ref_img_pil.size
    _, _, ref_inference_height, ref_inference_width = images[args.ref_image]['img'].shape

    scale_x = ref_inference_width / ref_original_width
    scale_y = ref_inference_height / ref_original_height

    query_point_inference = (query_point[0] * scale_x, query_point[1] * scale_y)

    print(f"  Query point (original): ({query_point[0]:.1f}, {query_point[1]:.1f})")
    print(f"  Query point (inference): ({query_point_inference[0]:.1f}, {query_point_inference[1]:.1f})")

    # Get dense 3D points for reference image
    pts3d_ref = pts3d_dense_all[args.ref_image].cpu().numpy()

    if args.min_height is not None:
        z_min, z_max = pts3d_ref[..., 2].min(), pts3d_ref[..., 2].max()
        print(f"  Height range in reference image: {z_min:.2f} to {z_max:.2f}")
        print(f"  Using minimum height threshold: {args.min_height}")

    query_pt3d = find_3d_point_from_2d(pts3d_ref, intrinsics_all[args.ref_image],
                                        cam2world_all[args.ref_image],
                                        query_point_inference,
                                        min_height=args.min_height,
                                        search_radius=20)

    if query_pt3d is None:
        print("  ERROR: Could not find valid 3D point at query location")
        print("  This may mean the point is in an area without valid 3D reconstruction")
        print("  Try a different point or use the pairwise matching script instead")
        return 1

    print(f"  Found 3D point: ({query_pt3d[0]:.3f}, {query_pt3d[1]:.3f}, {query_pt3d[2]:.3f})")

    # Create output directory
    output_dir = os.path.join(args.samples_dir, 'matches_global')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\\nOutput directory: {output_dir}")

    # Project 3D point to all other images
    print(f"\\nProjecting point to {len(image_files) - 1} target images...")
    print("=" * 80)

    results_summary = []

    for target_idx, target_img_path in enumerate(image_files):
        if target_idx == args.ref_image:
            continue  # Skip reference image

        print(f"\\nProcessing: Reference → Image {target_idx + 1}")
        print(f"Target: {os.path.basename(target_img_path)}")

        # Get target image dimensions
        target_img_pil = Image.open(target_img_path)
        target_original_width, target_original_height = target_img_pil.size
        _, _, target_inference_height, target_inference_width = images[target_idx]['img'].shape

        target_scale_x = target_inference_width / target_original_width
        target_scale_y = target_inference_height / target_original_height

        # Project 3D point to target image
        matched_point_inference = project_3d_to_2d(query_pt3d,
                                                   intrinsics_all[target_idx],
                                                   cam2world_all[target_idx])

        if matched_point_inference is None:
            print("  WARNING: Point is behind camera or outside view!")
            continue

        # Scale back to original resolution
        matched_point = (matched_point_inference[0] / target_scale_x,
                        matched_point_inference[1] / target_scale_y)

        print(f"  Matched point (inference): ({matched_point_inference[0]:.1f}, {matched_point_inference[1]:.1f})")
        print(f"  Matched point (original): ({matched_point[0]:.1f}, {matched_point[1]:.1f})")

        # Create visualization
        output_path = os.path.join(output_dir, f'ref{args.ref_image}_to_img{target_idx}.png')
        visualize_single_match(
            ref_img_path, target_img_path,
            query_point, matched_point,
            output_path, target_idx + 1,
            show_display=not args.no_display
        )

        results_summary.append({
            'target_idx': target_idx + 1,
            'target_name': os.path.basename(target_img_path),
            'matched_point': matched_point
        })

    # Print summary
    print("\\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Reference image: {os.path.basename(ref_img_path)} (index {args.ref_image})")
    print(f"Query point: ({query_point[0]:.1f}, {query_point[1]:.1f})")
    print(f"3D point (world): ({query_pt3d[0]:.3f}, {query_pt3d[1]:.3f}, {query_pt3d[2]:.3f})")
    print(f"\\nMatched points in target images:")
    for result in results_summary:
        mp = result['matched_point']
        print(f"  Image {result['target_idx']} ({result['target_name']}): " +
              f"({mp[0]:.1f}, {mp[1]:.1f})")

    print(f"\\nVisualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
