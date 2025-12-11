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
from matplotlib.widgets import RectangleSelector

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

    def collect_bbox_for_target(self, target_img_path, target_idx, total_targets):
        """
        Show target image and let user draw bounding box interactively.

        Args:
            target_img_path: Path to target image file
            target_idx: Index of this target image (1-indexed)
            total_targets: Total number of target images

        Returns:
            tuple: (x_min, y_min, x_max, y_max) in display coordinates, or None if skipped
        """
        # Load target image at original resolution
        target_img = np.array(Image.open(target_img_path))

        # Create new figure for bbox collection
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(target_img)
        ax.set_title(
            f'Target Image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}\n'
            f'CLICK AND DRAG to draw bounding box | ENTER: confirm | ESC: skip (use full image)',
            fontsize=12, fontweight='bold', pad=15, color='blue'
        )
        ax.axis('off')

        # State for bbox and confirmation
        bbox_data = {'bbox': None, 'selector': None}

        def on_select(eclick, erelease):
            """Called when rectangle is drawn/modified."""
            if eclick is None or erelease is None:
                return

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            if x1 is None or x2 is None or y1 is None or y2 is None:
                return

            # Normalize (min, max)
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            bbox_data['bbox'] = (x_min, y_min, x_max, y_max)

            # Show dimensions
            width = x_max - x_min
            height = y_max - y_min
            print(f"  Box drawn: ({x_min:.0f}, {y_min:.0f}) → ({x_max:.0f}, {y_max:.0f}) "
                  f"[{width:.0f}×{height:.0f}px] - Press ENTER to confirm")

            # Update title to show box is drawn
            ax.set_title(
                f'Target Image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}\n'
                f'Box: {width:.0f}×{height:.0f}px | ENTER: confirm | ESC: skip | Can drag corners to adjust',
                fontsize=12, fontweight='bold', pad=15, color='green'
            )
            fig.canvas.draw_idle()

        def on_key(event):
            """Handle keyboard confirmation."""
            if event.key == 'enter':
                if bbox_data['bbox'] is None:
                    print("  ⚠ No box drawn! Using full image.")
                else:
                    print("  ✓ Box confirmed!")
                plt.close(fig)
            elif event.key == 'escape':
                print("  ⊗ Skipped (using full image)")
                bbox_data['bbox'] = None
                plt.close(fig)

        # Create rectangle selector with more visible settings
        try:
            selector = RectangleSelector(
                ax, on_select,
                useblit=False,  # Disable blitting for better compatibility
                button=[1],  # Left mouse button only
                minspanx=20, minspany=20,  # Minimum box size
                spancoords='pixels',
                props=dict(
                    facecolor='red',
                    edgecolor='red',
                    alpha=0.3,
                    fill=True,
                    linewidth=3
                ),
                interactive=True,  # Allow dragging corners to adjust
                drag_from_anywhere=True,  # Allow dragging from inside the box
                handle_props=dict(
                    markersize=10,
                    markerfacecolor='red',
                    markeredgecolor='white',
                    markeredgewidth=2
                )
            )
            bbox_data['selector'] = selector
        except TypeError:
            # Older matplotlib version - use simpler parameters
            selector = RectangleSelector(
                ax, on_select,
                useblit=False,
                button=[1],
                minspanx=20, minspany=20,
                spancoords='pixels',
                rectprops=dict(
                    facecolor='red',
                    edgecolor='red',
                    alpha=0.3,
                    fill=True,
                    linewidth=3
                ),
                interactive=True
            )
            bbox_data['selector'] = selector

        # Connect keyboard handler
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Show modal dialog (blocks until closed)
        print(f"\n{'='*60}")
        print(f"→ Target image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}")
        print(f"   CLICK and DRAG to draw a bounding box")
        print(f"   Press ENTER when done, ESC to skip")
        print(f"{'='*60}")
        plt.show(block=True)

        # Keep reference to selector to prevent garbage collection
        _ = selector

        return bbox_data['bbox']

    def collect_multiple_bboxes_and_match(self, target_img_path, target_idx, total_targets,
                                          query_point, query_point_inference):
        """
        Multi-bbox mode: Allow drawing multiple bounding boxes, then process and show results inline.

        Args:
            target_img_path: Path to target image file
            target_idx: Index of this target image (1-indexed)
            total_targets: Total number of target images
            query_point: Query point in display coordinates
            query_point_inference: Query point in inference coordinates

        Returns:
            List of match results for this target image
        """
        # Load target image at original resolution
        target_img = np.array(Image.open(target_img_path))

        # Create new figure for multi-bbox collection
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(target_img)
        ax.set_title(
            f'Target Image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}\n'
            f'CLICK AND DRAG to draw bounding boxes (can draw multiple) | ENTER: process & show matches | ESC: skip',
            fontsize=12, fontweight='bold', pad=15, color='blue'
        )
        ax.axis('off')

        # State for multiple bboxes
        bbox_state = {
            'bboxes': [],  # List of completed bboxes
            'current_selector': None,
            'done': False,
            'skipped': False,
            'rect_patches': []  # Visual rectangles for completed boxes
        }

        def on_select(eclick, erelease):
            """Called when rectangle is drawn/modified."""
            if eclick is None or erelease is None:
                return

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            if x1 is None or x2 is None or y1 is None or y2 is None:
                return

            # Normalize (min, max)
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            width = x_max - x_min
            height = y_max - y_min

            if width < 20 or height < 20:
                return  # Too small

            # Add completed bbox to list
            new_bbox = (x_min, y_min, x_max, y_max)
            bbox_state['bboxes'].append(new_bbox)

            print(f"  Box {len(bbox_state['bboxes'])}: ({x_min:.0f}, {y_min:.0f}) → ({x_max:.0f}, {y_max:.0f}) [{width:.0f}×{height:.0f}px]", flush=True)

            # Draw a permanent rectangle for this bbox
            rect = plt.Rectangle((x_min, y_min), width, height,
                                 fill=False, edgecolor='yellow', linewidth=2, linestyle='-')
            ax.add_patch(rect)
            bbox_state['rect_patches'].append(rect)

            # Add bbox number label
            bbox_label = ax.text(x_min + 5, y_min + 20, f"#{len(bbox_state['bboxes'])}",
                                color='yellow', fontsize=14, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            bbox_state['rect_patches'].append(bbox_label)

            # Update title
            ax.set_title(
                f'Target Image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}\n'
                f'{len(bbox_state["bboxes"])} box(es) drawn | ENTER: process & show matches | ESC: skip | Draw more boxes...',
                fontsize=12, fontweight='bold', pad=15, color='green'
            )
            fig.canvas.draw_idle()

        def on_key(event):
            """Handle keyboard events."""
            print(f"  DEBUG: Key pressed: {event.key}")
            if event.key == 'enter':
                if len(bbox_state['bboxes']) == 0:
                    print("  ⚠ No boxes drawn! Skipping this image.")
                    bbox_state['skipped'] = True
                    bbox_state['done'] = False
                else:
                    print(f"  ✓ Ready to process {len(bbox_state['bboxes'])} box(es)...")
                    print(f"  DEBUG: Setting done=True, skipped=False")
                    bbox_state['done'] = True
                    bbox_state['skipped'] = False
                # DON'T close figure here - let event loop exit naturally
                print(f"  DEBUG: Flags set, returning from handler")
            elif event.key == 'escape':
                print("  ⊗ Skipped (no matching for this image)")
                bbox_state['skipped'] = True
                bbox_state['done'] = False
                # DON'T close figure here either
                print(f"  DEBUG: Skip flag set, returning from handler")

        # Create rectangle selector
        try:
            selector = RectangleSelector(
                ax, on_select,
                useblit=False,
                button=[1],
                minspanx=20, minspany=20,
                spancoords='pixels',
                props=dict(
                    facecolor='red',
                    edgecolor='red',
                    alpha=0.3,
                    fill=True,
                    linewidth=3
                ),
                interactive=False,  # Don't allow adjusting, just draw new ones
                drag_from_anywhere=False
            )
            bbox_state['current_selector'] = selector
        except TypeError:
            selector = RectangleSelector(
                ax, on_select,
                useblit=False,
                button=[1],
                minspanx=20, minspany=20,
                spancoords='pixels',
                rectprops=dict(
                    facecolor='red',
                    edgecolor='red',
                    alpha=0.3,
                    fill=True,
                    linewidth=3
                )
            )
            bbox_state['current_selector'] = selector

        # Connect keyboard handler
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Show modal dialog with manual event loop
        print(f"\n{'='*60}")
        print(f"→ Target image {target_idx}/{total_targets}: {os.path.basename(target_img_path)}")
        print(f"   Draw multiple bounding boxes (CLICK and DRAG for each)")
        print(f"   Press ENTER to process, ESC to skip")
        print(f"{'='*60}")

        plt.show(block=False)  # Non-blocking show
        print(f"  DEBUG: Window shown, entering event loop...")

        # Manual event loop - wait until done or skipped
        loop_count = 0
        while not bbox_state['done'] and not bbox_state['skipped']:
            plt.pause(0.1)  # Process events and wait 100ms
            loop_count += 1
            if loop_count % 10 == 0:  # Print every second
                print(f"  DEBUG: Still waiting... (done={bbox_state['done']}, skip={bbox_state['skipped']}, boxes={len(bbox_state['bboxes'])})")
            if not plt.fignum_exists(fig.number):
                # Window was closed manually
                print("  ⚠ Window closed manually, skipping...")
                bbox_state['skipped'] = True
                break

        print(f"  DEBUG: Exited event loop. done={bbox_state['done']}, skipped={bbox_state['skipped']}, bboxes={len(bbox_state['bboxes'])}")

        # Close the figure explicitly
        try:
            print(f"  DEBUG: Closing figure...")
            plt.close(fig)
            print(f"  DEBUG: Figure closed.")
        except Exception as e:
            print(f"  DEBUG: Error closing figure: {e}")

        # Debug: Check state after window closes
        print(f"\n  DEBUG: Ready to process. done={bbox_state['done']}, skipped={bbox_state['skipped']}, bboxes={len(bbox_state['bboxes'])}")

        # Check if skipped
        if bbox_state['skipped'] or not bbox_state['done']:
            print(f"  → Skipping processing (skipped={bbox_state['skipped']}, done={bbox_state['done']})")
            return []

        # Process matches for each bbox
        print(f"\n  → Processing {len(bbox_state['bboxes'])} bounding box(es) for matching...")

        # Load target image for inference
        target_images = load_images([target_img_path], size=self.image_size)
        target_img_data = target_images[0]

        # Get target dimensions
        target_img_pil = Image.open(target_img_path)
        target_original_width, target_original_height = target_img_pil.size
        _, _, target_inference_height, target_inference_width = target_img_data['img'].shape

        target_scale_x = target_inference_width / target_original_width
        target_scale_y = target_inference_height / target_original_height

        # Run inference once
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
            return []

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

        # Collect all matches from all bboxes, then find THE BEST one
        print(f"\n  → Finding best match across {len(bbox_state['bboxes'])} bounding box(es)...")

        best_match = None
        best_distance = float('inf')
        best_bbox_idx = None
        best_bbox = None

        for bbox_idx, bbox_display in enumerate(bbox_state['bboxes'], 1):
            print(f"    Checking Box #{bbox_idx}...")

            # Filter matches by this bbox
            matches_im0_filtered, matches_im1_filtered = self.filter_matches_by_bbox(
                matches_im0, matches_im1,
                bbox_display, target_scale_x, target_scale_y,
                target_inference_width, target_inference_height
            )

            if len(matches_im0_filtered) == 0:
                print(f"      No matches in Box #{bbox_idx}")
                continue

            # Find nearest match with height filtering in this box
            nearest_idx, distance = find_nearest_match(
                query_point_inference, matches_im0_filtered, matches_im1_filtered,
                pts3d_ref=pts3d_ref,
                pts3d_target=pts3d_target,
                min_height=height_threshold
            )

            matched_point = matches_im1_filtered[nearest_idx]

            # Scale matched point to original coordinates
            matched_x = matched_point[0] / target_scale_x
            matched_y = matched_point[1] / target_scale_y

            print(f"      Candidate: ({matched_x:.1f}, {matched_y:.1f}) | Distance: {distance:.2f}px")

            # Keep track of the best match across all boxes
            if distance < best_distance:
                best_distance = distance
                best_match = (matched_x, matched_y)
                best_bbox_idx = bbox_idx
                best_bbox = bbox_display

        # Create result for the single best match
        results = []
        if best_match is not None:
            print(f"\n  ✓ Best match found in Box #{best_bbox_idx}: {best_match} | Distance: {best_distance:.2f}px")
            results.append({
                'bbox_idx': best_bbox_idx,
                'bbox': best_bbox,
                'matched_point': best_match,
                'distance': best_distance
            })

            # Show result inline (pass all bboxes to show unselected ones too)
            self._show_inline_results(target_img, target_idx, query_point, results,
                                     os.path.basename(target_img_path), all_bboxes=bbox_state['bboxes'])
        else:
            print(f"\n  ⚠ No matches found in any bounding box!")

        return results

    def _show_inline_results(self, target_img, target_idx, query_point, results, img_name, all_bboxes=None):
        """Display matching results inline on the target image."""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(target_img)

        # Draw all bounding boxes in gray (not selected)
        if all_bboxes:
            for bbox_idx, bbox in enumerate(all_bboxes, 1):
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min

                # Check if this is the selected bbox
                is_selected = False
                if results and len(results) > 0:
                    if results[0]['bbox_idx'] == bbox_idx:
                        is_selected = True

                if not is_selected:
                    # Draw unselected bbox in gray
                    rect = plt.Rectangle((x_min, y_min), width, height,
                                         fill=False, edgecolor='gray', linewidth=2, linestyle=':', alpha=0.5)
                    ax.add_patch(rect)
                    # Draw bbox number in gray
                    ax.text(x_min + 5, y_min + 20, f"#{bbox_idx}",
                           color='gray', fontsize=12, alpha=0.6,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

        # Draw the selected bbox and its match
        for result in results:
            bbox = result['bbox']
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            # Draw selected bbox in green
            rect = plt.Rectangle((x_min, y_min), width, height,
                                 fill=False, edgecolor='green', linewidth=3, linestyle='-')
            ax.add_patch(rect)

            # Draw bbox number
            ax.text(x_min + 5, y_min + 20, f"#{result['bbox_idx']} ✓",
                   color='green', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

            # Draw match point
            mx, my = result['matched_point']
            ax.plot(mx, my, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)

            # Draw line from bbox center to match
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            ax.plot([cx, mx], [cy, my], 'r-', linewidth=2, alpha=0.6)

            # Add match info label
            ax.text(mx + 10, my - 10,
                   f"#{result['bbox_idx']}: d={result['distance']:.1f}px",
                   color='red', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        match_text = "BEST MATCH" if results and len(results) > 0 else "NO MATCH"
        ax.set_title(
            f'Target Image {target_idx}: {img_name}\n'
            f'{match_text} | Query point: ({query_point[0]:.0f}, {query_point[1]:.0f})',
            fontsize=14, fontweight='bold', pad=15, color='darkgreen' if results else 'red'
        )
        ax.axis('off')

        # Save result
        output_path = os.path.join(self.output_dir, f'bbox_matches_{target_idx}_{int(time.time())}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved results: {output_path}")

        # Show window
        plt.show(block=False)
        plt.pause(2)  # Show for 2 seconds
        plt.close()

    def filter_matches_by_bbox(self, matches_ref, matches_target,
                               bbox_display, target_scale_x, target_scale_y,
                               target_inference_width, target_inference_height):
        """
        Filter matches to only include those within the bounding box.

        CRITICAL: Bbox must be transformed from display to inference coordinates!

        Args:
            matches_ref: (N, 2) array in inference coordinates
            matches_target: (N, 2) array in inference coordinates
            bbox_display: (x_min, y_min, x_max, y_max) in display/original coordinates
            target_scale_x, target_scale_y: display → inference scale factors
            target_inference_width, target_inference_height: inference image dimensions

        Returns:
            Filtered (matches_ref, matches_target) or originals if no matches in bbox
        """
        if bbox_display is None:
            return matches_ref, matches_target

        # Transform bbox to inference coordinates
        x_min_disp, y_min_disp, x_max_disp, y_max_disp = bbox_display
        x_min = max(0, x_min_disp * target_scale_x)
        y_min = max(0, y_min_disp * target_scale_y)
        x_max = min(target_inference_width, x_max_disp * target_scale_x)
        y_max = min(target_inference_height, y_max_disp * target_scale_y)

        # Filter matches in target image (inference coordinates)
        in_bbox = (
            (matches_target[:, 0] >= x_min) &
            (matches_target[:, 0] <= x_max) &
            (matches_target[:, 1] >= y_min) &
            (matches_target[:, 1] <= y_max)
        )

        n_in_bbox = np.sum(in_bbox)
        if n_in_bbox == 0:
            print(f"  WARNING: No matches inside bbox! Using all {len(matches_target)} matches.")
            return matches_ref, matches_target

        print(f"  Filtered to {n_in_bbox}/{len(matches_target)} matches inside bbox")
        return matches_ref[in_bbox], matches_target[in_bbox]

    def _process_points(self):
        """Background thread to process point tracking requests."""
        while not self.should_exit:
            try:
                # Get next point from queue (with timeout to check should_exit)
                try:
                    query_data = self.point_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self.processing = True

                # Extract point and bboxes from query_data
                query_point = query_data['point']
                bboxes = query_data['bboxes']

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
                for i, target_idx in enumerate(range(1, len(self.image_files))):
                    target_img_path = self.image_files[target_idx]
                    print(f"\nImage {target_idx}/{len(self.image_files)-1}: {os.path.basename(target_img_path)}")

                    # Get pre-collected bounding box for this target
                    bbox_display = bboxes[i]
                    if bbox_display is not None:
                        x_min, y_min, x_max, y_max = bbox_display
                        print(f"  Using bbox: ({x_min:.0f}, {y_min:.0f}) → ({x_max:.0f}, {y_max:.0f})")
                    else:
                        print(f"  Using full image (no bbox)")

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

                    # Filter matches by bounding box (NEW)
                    matches_im0, matches_im1 = self.filter_matches_by_bbox(
                        matches_im0, matches_im1,
                        bbox_display, target_scale_x, target_scale_y,
                        target_inference_width, target_inference_height
                    )

                    if len(matches_im0) == 0:
                        print("  No matches after bbox filtering!")
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
                        'scale_y': target_scale_y,
                        'bbox': bbox_display  # Store bbox in display coordinates
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

        # Create a figure showing reference + all matches (2 columns layout)
        n_targets = len(self.last_results)
        n_rows = (n_targets + 1) // 2  # Calculate rows needed for 2 columns
        fig = plt.figure(figsize=(40, 10 * n_rows))
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.2)

        for i, result in enumerate(self.last_results):
            ax = fig.add_subplot(gs[i // 2, i % 2])

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

            # Draw bounding box if it was used
            if result.get('bbox') is not None:
                bbox = result['bbox']
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                # Draw rectangle on target image (offset by W0 for combined image)
                rect = plt.Rectangle((x_min + W0, y_min), width, height,
                                     fill=False, edgecolor='green', linewidth=2, linestyle='--')
                ax.add_patch(rect)

                title_text = (f"Image {result['target_idx']}: ({x1:.0f}, {y1:.0f}) | "
                             f"dist: {result['distance']:.1f}px | bbox: {width:.0f}×{height:.0f}px")
            else:
                title_text = (f"Image {result['target_idx']}: ({x1:.0f}, {y1:.0f}) | "
                             f"dist: {result['distance']:.1f}px | full image")

            ax.set_title(title_text, fontsize=10)
            ax.axis('off')

        plt.suptitle(f"Point Tracking Results: ({self.current_point[0]:.1f}, {self.current_point[1]:.1f})",
                    fontsize=14, fontweight='bold')

        # Save to file with higher DPI for better quality at larger size
        output_path = os.path.join(self.output_dir, f'tracking_{int(time.time())}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization: {output_path}")
        plt.close()

        # Open in viewnior
        import subprocess
        try:
            subprocess.Popen(['viewnior', output_path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            print(f"Opened in viewnior: {output_path}")
        except FileNotFoundError:
            print("viewnior not found. Install it with: sudo pacman -S viewnior")
        except Exception as e:
            print(f"Could not open in viewnior: {e}")

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes and event.button == 1:  # Left click
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(f"\n→ Clicked: ({x:.1f}, {y:.1f})")

                # Update reference image to show clicked point
                if hasattr(self, 'ref_point_marker'):
                    self.ref_point_marker.set_data([x], [y])
                else:
                    self.ref_point_marker, = self.ref_ax.plot(x, y, 'r*',
                                                               markersize=15,
                                                               markeredgecolor='white',
                                                               markeredgewidth=2)
                self.ref_fig.canvas.draw_idle()

                # Route to appropriate workflow based on mode
                if self.args.bbox_mode:
                    # Multi-bbox mode: Draw multiple boxes per image, show inline results
                    print("\n" + "="*60)
                    print("MULTI-BBOX MODE: Draw multiple boxes per target image")
                    print("="*60)

                    query_point = (x, y)
                    query_point_inference = (
                        x * self.scale_x,
                        y * self.scale_y
                    )

                    # Process each target image with multi-bbox workflow
                    for target_idx in range(1, len(self.image_files)):
                        target_img_path = self.image_files[target_idx]
                        results = self.collect_multiple_bboxes_and_match(
                            target_img_path, target_idx, len(self.image_files) - 1,
                            query_point, query_point_inference
                        )

                    print("\n" + "="*60)
                    print("MULTI-BBOX PROCESSING COMPLETE!")
                    print("="*60)

                else:
                    # Original workflow: Single bbox per image, batch collection
                    print("\n" + "="*60)
                    print("SINGLE-BBOX MODE: Collecting bounding boxes for all target images")
                    print("="*60)
                    bboxes = []
                    for target_idx in range(1, len(self.image_files)):
                        target_img_path = self.image_files[target_idx]
                        bbox = self.collect_bbox_for_target(
                            target_img_path, target_idx, len(self.image_files) - 1
                        )
                        bboxes.append(bbox)

                    print("\n" + "="*60)
                    print("BBOX COLLECTION COMPLETE - Starting processing...")
                    print("="*60)

                    # Queue point with bboxes attached
                    self.point_queue.put({
                        'point': (x, y),
                        'bboxes': bboxes
                    })

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
        print("  • Results will open in viewnior image viewer")
        print("  • Press 'q' to exit the application")
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
    parser.add_argument('--bbox-mode', action='store_true',
                       help='Enable multi-bbox mode: draw multiple bounding boxes per target image with inline results')
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
