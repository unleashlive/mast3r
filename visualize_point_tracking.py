#!/usr/bin/env python3
"""
Interactive Point Tracking Visualization for MASt3R

This script allows you to:
1. Click on points in a reference image interactively
2. Track each point across all other images using MASt3R matching
3. Visualize the matches in real-time with automatic updates

Usage:
    python visualize_point_tracking.py [--device cuda|cpu] [--image_size 512]

    Click on the reference image to probe points. Results appear automatically.
    Press 'q' or close the window to exit.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image
import threading
import queue
import time

# Prevent Qt/OpenCV conflicts
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# Set interactive backend for matplotlib
import matplotlib
# Try to use an interactive backend (prefer TkAgg to avoid Qt/cv2 conflicts)
_backend_loaded = False
_backend_error = None
for backend in ['TkAgg', 'GTK3Agg', 'WXAgg']:
    try:
        matplotlib.use(backend, force=True)
        _backend_loaded = True
        print(f"Using matplotlib backend: {backend}")
        break
    except Exception as e:
        _backend_error = str(e)
        continue

if not _backend_loaded:
    print(f"Warning: Could not load preferred interactive backends")
    if _backend_error:
        print(f"  Last error: {_backend_error}")
    print("  Falling back to Agg (non-interactive) - visualizations will be saved but not displayed")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add MASt3R paths
import mast3r.utils.path_to_dust3r  # noqa

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images


class InteractivePointTracker:
    """Interactive point tracking system with continuous click support."""

    def __init__(self, model, device, image_files, image_size, args):
        self.model = model
        self.device = device
        self.image_files = image_files
        self.image_size = image_size
        self.args = args

        self.ref_img_path = image_files[0]
        self.ref_img = np.array(Image.open(self.ref_img_path))

        # Load and cache reference image data
        print("Loading reference image for inference...")
        ref_images = load_images([self.ref_img_path], size=image_size)
        self.ref_img_data = ref_images[0]

        # Get coordinate transformation parameters
        ref_img_pil = Image.open(self.ref_img_path)
        ref_original_width, ref_original_height = ref_img_pil.size
        _, _, ref_inference_height, ref_inference_width = self.ref_img_data['img'].shape

        self.scale_x = ref_inference_width / ref_original_width
        self.scale_y = ref_inference_height / ref_original_height

        print(f"Coordinate transform: {ref_original_width}x{ref_original_height} → "
              f"{ref_inference_width}x{ref_inference_height} "
              f"(scale: {self.scale_x:.4f}, {self.scale_y:.4f})")

        # Processing queue and state
        self.point_queue = queue.Queue()
        self.processing = False
        self.should_exit = False
        self.current_point = None
        self.last_results = []

        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(self.ref_img_path), 'matches')
        os.makedirs(self.output_dir, exist_ok=True)

        # Start processing thread
        self.processor_thread = threading.Thread(target=self._process_points, daemon=True)
        self.processor_thread.start()

    def _process_points(self):
        """Background thread to process point tracking requests."""
        while not self.should_exit:
            try:
                # Get next point from queue (with timeout to check should_exit)
                try:
                    query_point = self.point_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self.processing = True
                self.current_point = query_point

                print(f"\n{'='*60}")
                print(f"Processing point: ({query_point[0]:.1f}, {query_point[1]:.1f})")
                print(f"{'='*60}")

                # Transform to inference coordinates
                query_point_inference = (
                    query_point[0] * self.scale_x,
                    query_point[1] * self.scale_y
                )

                results = []

                # Track across all target images
                for target_idx in range(1, len(self.image_files)):
                    target_img_path = self.image_files[target_idx]
                    print(f"\nImage {target_idx}/{len(self.image_files)-1}: {os.path.basename(target_img_path)}")

                    # Load target image
                    target_images = load_images([target_img_path], size=self.image_size)
                    target_img_data = target_images[0]

                    # Get target dimensions
                    target_img_pil = Image.open(target_img_path)
                    target_original_width, target_original_height = target_img_pil.size
                    _, _, target_inference_height, target_inference_width = target_img_data['img'].shape

                    target_scale_x = target_inference_width / target_original_width
                    target_scale_y = target_inference_height / target_original_height

                    # Run inference
                    with torch.no_grad():
                        output = inference([(self.ref_img_data, target_img_data)], self.model,
                                         self.device, batch_size=1, verbose=False)

                    # Extract descriptors and 3D points
                    pred1, pred2 = output['pred1'], output['pred2']
                    desc1 = pred1['desc'].squeeze(0).detach()
                    desc2 = pred2['desc'].squeeze(0).detach()

                    pts3d_ref = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
                    pts3d_target = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

                    # Find matches
                    matches_im0, matches_im1 = fast_reciprocal_NNs(
                        desc1, desc2,
                        subsample_or_initxy1=8,
                        device=self.device,
                        dist='dot',
                        block_size=2**13
                    )

                    if isinstance(matches_im0, torch.Tensor):
                        matches_im0 = matches_im0.cpu().numpy()
                    if isinstance(matches_im1, torch.Tensor):
                        matches_im1 = matches_im1.cpu().numpy()

                    if len(matches_im0) == 0:
                        print("  No matches found!")
                        continue

                    # Get query point height for filtering
                    query_z = None
                    qx, qy = int(round(query_point_inference[0])), int(round(query_point_inference[1]))
                    if 0 <= qy < pts3d_ref.shape[0] and 0 <= qx < pts3d_ref.shape[1]:
                        query_z = pts3d_ref[qy, qx, 2]

                    # Determine height threshold
                    height_threshold = None
                    if self.args.min_height is not None:
                        height_threshold = self.args.min_height
                    elif query_z is not None and np.isfinite(query_z):
                        height_threshold = query_z - self.args.height_tolerance

                    # Find nearest match
                    nearest_idx, distance = find_nearest_match(
                        query_point_inference, matches_im0, matches_im1,
                        pts3d_ref=pts3d_ref,
                        pts3d_target=pts3d_target,
                        min_height=height_threshold
                    )

                    matched_point = matches_im1[nearest_idx]

                    # Scale matched point to original coordinates
                    matched_x = matched_point[0] / target_scale_x
                    matched_y = matched_point[1] / target_scale_y

                    print(f"  Match: ({matched_x:.1f}, {matched_y:.1f}) | Distance: {distance:.2f}px")

                    results.append({
                        'target_idx': target_idx,
                        'target_path': target_img_path,
                        'matched_point': (matched_x, matched_y),
                        'distance': distance,
                        'scale_x': target_scale_x,
                        'scale_y': target_scale_y
                    })

                self.last_results = results
                self.processing = False

                # Update visualization
                self._update_visualization()

            except Exception as e:
                print(f"Error processing point: {e}")
                import traceback
                traceback.print_exc()
                self.processing = False

    def _update_visualization(self):
        """Update the visualization with current results."""
        if not self.current_point or not self.last_results:
            return

        # Create a figure showing reference + all matches
        n_targets = len(self.last_results)
        fig = plt.figure(figsize=(20, 4 * ((n_targets + 2) // 3)))
        gs = GridSpec(((n_targets + 2) // 3), 3, figure=fig, hspace=0.3, wspace=0.2)

        for i, result in enumerate(self.last_results):
            ax = fig.add_subplot(gs[i // 3, i % 3])

            # Load images
            img_ref = self.ref_img
            img_target = np.array(Image.open(result['target_path']))

            # Pad to same height
            H0, W0 = img_ref.shape[:2]
            H1, W1 = img_target.shape[:2]

            img_ref_pad = np.pad(img_ref, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                                'constant', constant_values=0)
            img_target_pad = np.pad(img_target, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                                   'constant', constant_values=0)

            img_combined = np.concatenate((img_ref_pad, img_target_pad), axis=1)

            ax.imshow(img_combined)

            # Draw match
            x0, y0 = self.current_point
            x1, y1 = result['matched_point']

            ax.plot([x0, x1 + W0], [y0, y1], '-o',
                   color='red', linewidth=2, markersize=8, alpha=0.8)
            ax.plot(x0, y0, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)

            ax.set_title(f"Image {result['target_idx']}: ({x1:.0f}, {y1:.0f}) | "
                        f"dist: {result['distance']:.1f}px", fontsize=10)
            ax.axis('off')

        plt.suptitle(f"Point Tracking Results: ({self.current_point[0]:.1f}, {self.current_point[1]:.1f})",
                    fontsize=14, fontweight='bold')

        # Save to file
        output_path = os.path.join(self.output_dir, f'tracking_{int(time.time())}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\nSaved visualization: {output_path}")
        plt.close()

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes and event.button == 1:  # Left click
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(f"\n→ Clicked: ({x:.1f}, {y:.1f})")
                self.point_queue.put((x, y))

                # Update reference image to show clicked point
                if hasattr(self, 'ref_point_marker'):
                    self.ref_point_marker.set_data([x], [y])
                else:
                    self.ref_point_marker, = self.ref_ax.plot(x, y, 'r*',
                                                               markersize=15,
                                                               markeredgecolor='white',
                                                               markeredgewidth=2)
                self.ref_fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'q':
            print("\nExiting...")
            self.should_exit = True
            plt.close('all')

    def run(self):
        """Run the interactive visualization."""
        backend = matplotlib.get_backend()

        # Check if we have an interactive backend
        if backend.lower() == 'agg':
            print("\n" + "="*60)
            print("ERROR: Interactive mode not available")
            print("="*60)
            print("No interactive matplotlib backend could be loaded.")
            print("\nTo fix this, install one of the following:")
            print("  - TkAgg: pip install tk")
            print("  - GTK3Agg: pip install pygobject")
            print("  - WXAgg: pip install wxPython")
            print("\nAlternatively, you can use the non-interactive script:")
            print("  python visualize_point_tracking.py --point x,y")
            print("="*60)
            return

        print("\n" + "="*60)
        print("INTERACTIVE POINT TRACKING")
        print("="*60)
        print("Instructions:")
        print("  • Click on the reference image to track a point")
        print("  • Results will be processed and displayed automatically")
        print("  • Press 'q' or close window to exit")
        print("="*60 + "\n")

        # Create reference image window
        try:
            self.ref_fig, self.ref_ax = plt.subplots(figsize=(12, 8))
            self.ref_ax.imshow(self.ref_img)
            self.ref_ax.set_title('Reference Image - Click to Track Points\n(Press "q" to exit)',
                                 fontsize=14, fontweight='bold')
            self.ref_ax.axis('off')

            # Connect event handlers
            self.ref_fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.ref_fig.canvas.mpl_connect('key_press_event', self.on_key)

            plt.show()
        except Exception as e:
            print(f"\nError displaying interactive window: {e}")
            print("The interactive mode requires a display server (X11, Wayland, etc.)")
        finally:
            # Wait for processing to finish
            self.should_exit = True
            if self.processor_thread.is_alive():
                self.processor_thread.join(timeout=2.0)


def find_nearest_match(query_point, matches_ref, matches_target,
                       pts3d_ref=None, pts3d_target=None,
                       min_height=None, max_height=None):
    """
    Find the index of the match nearest to the query point.

    Args:
        query_point: (x, y) tuple of query point coordinates
        matches_ref: (N, 2) array of match coordinates in reference image
        matches_target: (N, 2) array of match coordinates in target image
        pts3d_ref: Optional (H, W, 3) array of 3D points for reference image
        pts3d_target: Optional (H, W, 3) array of 3D points for target image
        min_height: Optional minimum Z-coordinate threshold for filtering
        max_height: Optional maximum Z-coordinate threshold for filtering

    Returns:
        Index of nearest match and distance
    """
    query_array = np.array(query_point)
    distances = np.linalg.norm(matches_ref - query_array, axis=1)

    # If height filtering is enabled, filter matches by Z-coordinate
    if (pts3d_ref is not None or pts3d_target is not None) and (min_height is not None or max_height is not None):
        valid_mask = np.ones(len(matches_ref), dtype=bool)
        filtered_by_ref = 0
        filtered_by_target = 0

        for i in range(len(matches_ref)):
            # Check reference image height
            if pts3d_ref is not None:
                x_idx = int(round(matches_ref[i, 0]))
                y_idx = int(round(matches_ref[i, 1]))

                if 0 <= y_idx < pts3d_ref.shape[0] and 0 <= x_idx < pts3d_ref.shape[1]:
                    z_coord = pts3d_ref[y_idx, x_idx, 2]
                    if min_height is not None and z_coord < min_height:
                        valid_mask[i] = False
                        filtered_by_ref += 1
                        continue
                    if max_height is not None and z_coord > max_height:
                        valid_mask[i] = False
                        filtered_by_ref += 1
                        continue

            # Check target image height
            if pts3d_target is not None and valid_mask[i]:
                x_idx = int(round(matches_target[i, 0]))
                y_idx = int(round(matches_target[i, 1]))

                if 0 <= y_idx < pts3d_target.shape[0] and 0 <= x_idx < pts3d_target.shape[1]:
                    z_coord = pts3d_target[y_idx, x_idx, 2]
                    if min_height is not None and z_coord < min_height:
                        valid_mask[i] = False
                        filtered_by_target += 1
                        continue
                    if max_height is not None and z_coord > max_height:
                        valid_mask[i] = False
                        filtered_by_target += 1
                        continue

        # Filter matches by height
        if not np.any(valid_mask):
            print(f"  WARNING: No matches in height range. Using all matches.")
        else:
            n_filtered = np.sum(~valid_mask)
            if n_filtered > 0:
                print(f"  Filtered {n_filtered} matches outside height range (ref: {filtered_by_ref}, target: {filtered_by_target})")
            distances[~valid_mask] = np.inf

    nearest_idx = np.argmin(distances)
    return nearest_idx, distances[nearest_idx]


def main():
    parser = argparse.ArgumentParser(description='Interactive point tracking across multiple images using MASt3R')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for model inference')
    parser.add_argument('--samples_dir', type=str,
                       default='/home/mordka/UNLEASH/work/mast3r/samples',
                       help='Directory containing sample images')
    parser.add_argument('--min-height', type=float, default=None,
                       help='Minimum relative height (Z-coordinate) for filtering matches. Use this to prefer elevated points (e.g., power lines) over ground points.')
    parser.add_argument('--height-tolerance', type=float, default=0.5,
                       help='Height tolerance around query point (default: 0.5). Only match points within this Z range of the query point height.')
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

    print(f"\nReference image: {os.path.basename(image_files[0])}")

    # Create and run interactive tracker
    try:
        tracker = InteractivePointTracker(model, args.device, image_files, args.image_size, args)
        tracker.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
