#!/usr/bin/env python3
"""
Interactive Point Tracking Visualization for MASt3R

This script allows you to:
1. Click on a point in a reference image
2. Track that point across all other images using MASt3R matching
3. Visualize the matches with colored lines connecting corresponding points

Usage:
    python visualize_point_tracking.py [--device cuda|cpu] [--image_size 512]
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch

# Set matplotlib backend before importing pyplot
import matplotlib
# Use Agg backend for non-interactive use, will switch to interactive if needed
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Add MASt3R paths
import mast3r.utils.path_to_dust3r  # noqa

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images


def select_point_interactively(image_path):
    """
    Display an image and allow user to click to select a point.

    Args:
        image_path: Path to the image file

    Returns:
        (x, y): Selected point coordinates, or None if cancelled
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array)
    ax.set_title('Click on a point to track across images\n(Close window to use clicked point)')
    ax.axis('off')

    print("\nPlease click on a point in the image...")
    print("Close the window when done.")

    points = plt.ginput(n=1, timeout=0)
    plt.close()

    if not points:
        print("No point selected!")
        return None

    x, y = points[0]
    print(f"Selected point: ({x:.1f}, {y:.1f})")

    return (float(x), float(y))


def find_nearest_match(query_point, matches_ref):
    """
    Find the index of the match nearest to the query point.

    Args:
        query_point: (x, y) tuple of query point coordinates
        matches_ref: (N, 2) array of match coordinates in reference image

    Returns:
        Index of nearest match
    """
    query_array = np.array(query_point)
    distances = np.linalg.norm(matches_ref - query_array, axis=1)
    nearest_idx = np.argmin(distances)
    return nearest_idx, distances[nearest_idx]


def sample_context_matches(matches_ref, matches_target, query_idx, n_context=19):
    """
    Sample additional context matches around the query match.

    Args:
        matches_ref: (N, 2) array of matches in reference image
        matches_target: (N, 2) array of matches in target image
        query_idx: Index of the query match
        n_context: Number of additional context matches to include

    Returns:
        Tuple of (selected_ref_matches, selected_target_matches, is_query_mask)
    """
    n_total = len(matches_ref)

    if n_total <= n_context + 1:
        # Include all matches
        is_query = np.zeros(n_total, dtype=bool)
        is_query[query_idx] = True
        return matches_ref, matches_target, is_query

    # Sample context matches
    # Strategy: sample evenly across all matches
    all_indices = np.arange(n_total)
    other_indices = np.delete(all_indices, query_idx)

    # Sample evenly spaced indices
    step = len(other_indices) // n_context
    if step < 1:
        step = 1
    sampled_indices = other_indices[::step][:n_context]

    # Combine query and context
    selected_indices = np.concatenate([[query_idx], sampled_indices])
    selected_indices = np.sort(selected_indices)

    is_query = np.zeros(len(selected_indices), dtype=bool)
    is_query[np.where(selected_indices == query_idx)[0][0]] = True

    return matches_ref[selected_indices], matches_target[selected_indices], is_query


def visualize_matches(ref_img_path, target_img_path, matches_ref, matches_target,
                     is_query, query_point, output_path, target_idx, show_display=True):
    """
    Create and save a visualization of matches between two images.

    Args:
        ref_img_path: Path to reference image
        target_img_path: Path to target image
        matches_ref: (N, 2) array of match coordinates in reference image
        matches_target: (N, 2) array of match coordinates in target image
        is_query: (N,) boolean array indicating which match is the query
        query_point: Original query point coordinates
        output_path: Path to save visualization
        target_idx: Index of target image (for title)
        show_display: Whether to display the plot interactively
    """
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

    # Get colormap
    cmap = plt.get_cmap('jet')
    n_matches = len(matches_ref)

    # Draw matches
    for i in range(n_matches):
        x0, y0 = matches_ref[i]
        x1, y1 = matches_target[i]

        if is_query[i]:
            # Draw query match in red with thicker line
            color = 'red'
            linewidth = 3
            markersize = 12
            alpha = 1.0
        else:
            # Context matches in gradient colors
            color = cmap(i / (n_matches - 1))
            linewidth = 1.5
            markersize = 8
            alpha = 0.7

        # Draw line from ref image to target image
        # Target image is offset by W0 horizontally
        ax.plot([x0, x1 + W0], [y0, y1], '-+',
                color=color, linewidth=linewidth, markersize=markersize,
                alpha=alpha, scalex=False, scaley=False)

    # Add query point marker in reference image
    ax.plot(query_point[0], query_point[1], 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)

    # Find which match is the query
    query_idx = np.where(is_query)[0][0]
    matched_point = matches_target[query_idx]

    ax.set_title(f'Point Tracking: Reference → Image {target_idx}\n' +
                f'Query Point: ({query_point[0]:.1f}, {query_point[1]:.1f}) → ' +
                f'Matched Point: ({matched_point[0]:.1f}, {matched_point[1]:.1f})\n' +
                f'Total Matches Shown: {n_matches}',
                fontsize=14, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Display interactively if requested
    if show_display:
        try:
            plt.show()
        except:
            pass  # Ignore display errors
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Track a point across multiple images using MASt3R')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for model inference')
    parser.add_argument('--samples_dir', type=str,
                       default='/home/mordka/UNLEASH/work/mast3r/samples',
                       help='Directory containing sample images')
    parser.add_argument('--point', type=str, default=None,
                       help='Query point coordinates as "x,y" (e.g., "1500,2000"). If not provided, will attempt interactive selection.')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display visualizations interactively, only save to disk')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("MASt3R Interactive Point Tracking Visualization")
    print("=" * 80)

    # Find all images in samples directory
    image_patterns = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(args.samples_dir, pattern)))

    image_files = sorted(image_files)

    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {args.samples_dir}")
        return 1

    print(f"\nFound {len(image_files)} images:")
    for i, img_path in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(img_path)}")

    # Load MASt3R model
    print(f"\nLoading MASt3R model on {args.device}...")
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(args.device)
    model.eval()
    print("Model loaded successfully!")

    # Select reference image (first image)
    ref_img_path = image_files[0]
    print(f"\nReference image: {os.path.basename(ref_img_path)}")

    # Get query point from command line or interactive selection
    if args.point:
        # Parse coordinates from command line
        try:
            x, y = map(float, args.point.split(','))
            query_point = (x, y)
            print(f"\nUsing query point from command line: ({x:.1f}, {y:.1f})")
        except ValueError:
            print(f"Error: Invalid point format '{args.point}'. Expected format: 'x,y' (e.g., '1500,2000')")
            return 1
    else:
        # Interactive point selection
        try:
            query_point = select_point_interactively(ref_img_path)
            if query_point is None:
                print("Point selection cancelled.")
                return 1
        except Exception as e:
            print(f"\nError with interactive selection: {e}")
            print("\nTip: Use --point x,y to specify coordinates directly")
            print("Example: python visualize_point_tracking.py --point 2000,1500")
            return 1

    # Create output directory
    output_dir = os.path.join(args.samples_dir, 'matches')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load reference image for inference
    print("\nLoading reference image for inference...")
    ref_images = load_images([ref_img_path], size=args.image_size)
    ref_img_data = ref_images[0]

    # Track point across all other images
    print(f"\nTracking point across {len(image_files) - 1} target images...")
    print("=" * 80)

    results_summary = []

    for target_idx in range(1, len(image_files)):
        target_img_path = image_files[target_idx]
        print(f"\nProcessing pair: Reference → Image {target_idx + 1}")
        print(f"Target: {os.path.basename(target_img_path)}")

        # Load target image
        target_images = load_images([target_img_path], size=args.image_size)
        target_img_data = target_images[0]

        # Run MASt3R inference
        print("  Running inference...")
        with torch.no_grad():
            output = inference([(ref_img_data, target_img_data)], model,
                             args.device, batch_size=1, verbose=False)

        # Extract descriptors
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()

        # Find matches using reciprocal nearest neighbors
        print("  Finding matches...")
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2,
            subsample_or_initxy1=8,
            device=args.device,
            dist='dot',
            block_size=2**13
        )

        # Convert to numpy if needed
        if isinstance(matches_im0, torch.Tensor):
            matches_im0 = matches_im0.cpu().numpy()
        if isinstance(matches_im1, torch.Tensor):
            matches_im1 = matches_im1.cpu().numpy()

        print(f"  Found {len(matches_im0)} total matches")

        if len(matches_im0) == 0:
            print("  WARNING: No matches found for this pair!")
            continue

        # Find nearest match to query point
        nearest_idx, distance = find_nearest_match(query_point, matches_im0)
        print(f"  Nearest match distance: {distance:.2f} pixels")

        matched_point = matches_im1[nearest_idx]
        print(f"  Query point ({query_point[0]:.1f}, {query_point[1]:.1f}) → " +
              f"Matched point ({matched_point[0]:.1f}, {matched_point[1]:.1f})")

        # Sample context matches
        selected_ref, selected_target, is_query = sample_context_matches(
            matches_im0, matches_im1, nearest_idx, n_context=19
        )

        print(f"  Visualizing {len(selected_ref)} matches (1 query + {len(selected_ref)-1} context)")

        # Create visualization
        output_path = os.path.join(output_dir, f'ref_to_img{target_idx + 1}.png')
        visualize_matches(
            ref_img_path, target_img_path,
            selected_ref, selected_target,
            is_query, query_point,
            output_path, target_idx + 1,
            show_display=not args.no_display
        )

        # Store results
        results_summary.append({
            'target_idx': target_idx + 1,
            'target_name': os.path.basename(target_img_path),
            'matched_point': matched_point,
            'distance': distance,
            'total_matches': len(matches_im0)
        })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Query point in reference image: ({query_point[0]:.1f}, {query_point[1]:.1f})")
    print(f"\nMatched points in target images:")
    for result in results_summary:
        mp = result['matched_point']
        print(f"  Image {result['target_idx']} ({result['target_name']}): " +
              f"({mp[0]:.1f}, {mp[1]:.1f}) | " +
              f"Distance: {result['distance']:.2f}px | " +
              f"Total matches: {result['total_matches']}")

    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
