"""
Title: Pulsating Aurora Auto-Detection Module
Author: James Hui
Affiliation: River Hill High School, UMD
Email: james.y.hui@gmail.com
GitHub: https://github.com/James-Tiny-Tjib
Date Created: 2026-01-06
Last Updated: 2026-01-08
"""

import os
import re
from datetime import datetime, timedelta
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
import itertools
import random
import matplotlib.animation as animation

# --- UTILITY AND DATA FUNCTIONS ----------------------------------------------------------------------------------

def generate_grid_regions(grid_dims, image_size, box_size):
    # Goal: Generate the top-left coordinates for an evenly spaced grid of regions.
    # Input: grid_dims (tuple), image_size (tuple), box_size (tuple)
    # Output: A list of (y, x) tuples for the top-left corner of each grid region (list of tuples).
    grid_rows, grid_cols = grid_dims
    image_height, image_width = image_size
    box_height, box_width = box_size
    
    regions = []
    # Calculate spacing to center the grid
    x_spacing = image_width / (grid_cols + 1)
    y_spacing = image_height / (grid_rows + 1)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            center_x = (c + 1) * x_spacing
            center_y = (r + 1) * y_spacing
            top_left_x = int(center_x - (box_width / 2))
            top_left_y = int(center_y - (box_height / 2))
            regions.append((top_left_y, top_left_x))
            
    return regions

def load_frames(start_idx, end_idx, frame_map):
    # Goal: Load a specific range of image frames, potentially spanning multiple TIFF files.
    # Input: start_idx (int), end_idx (int), frame_map (list of dicts)
    # Output: A 3D NumPy array of the loaded frames, with dimensions (frame, height, width).
    frames_list = []
    for f_info in frame_map:
        file_start, file_end = f_info['start_frame'], f_info['start_frame'] + f_info['count']
        if start_idx < file_end and end_idx > file_start:
            load_start = max(start_idx, file_start) - file_start
            load_end = min(end_idx, file_end) - file_start
            with tifffile.TiffFile(f_info['path']) as tif:
                frames_list.append(tif.asarray(key=slice(load_start, load_end)))
    if not frames_list: return np.array([])
    return np.concatenate(frames_list, axis=0)

def parse_pulsation_report(report_path):
    # Goal: Parse a pulsation report file to extract event data and grid dimensions.
    # Input: report_path (str)
    # Output: A tuple containing (intervals, grid_dims).
    #         - intervals: A list of dictionaries, each representing a detected event.
    #         - grid_dims: A tuple (rows, cols) defining the analysis grid size.
    with open(report_path, 'r') as f: lines = f.readlines()
    
    date_str, intervals = "", []
    grid_dims = None
    pixel_size = 6  # Default fallback
    
    grid_pattern = re.compile(r'\((\d+)x(\d+)\)')
    pixel_pattern = re.compile(r'Pixel Grid Size: (\d+)x') # New pattern

    for line in lines:
        line = line.strip() # Clean whitespace
        if not line: continue 

        if line.startswith("Date:"): 
            date_str = line.split("Date:")[1].strip()
        elif "Pixel Grid Size:" in line:
            # Parse the 'n' from 'nxn'
            match = pixel_pattern.search(line)
            if match:
                pixel_size = int(match.group(1))
        elif "Region Reference Grid" in line:
            match = grid_pattern.search(line)
            if match:
                grid_dims = (int(match.group(1)), int(match.group(2)))

        parts = line.split()
        # Expect at least 11 columns. 
        # Check if first part looks like a timestamp (HH:MM:SS.ms)
        if len(parts) >= 11 and ":" in parts[0] and "." in parts[0] and "Time" not in parts[0]:
            try:
                t1_dt = datetime.strptime(f"{date_str} {parts[0]}", "%Y-%m-%d %H:%M:%S.%f")
                t2_dt = datetime.strptime(f"{date_str} {parts[1]}", "%Y-%m-%d %H:%M:%S.%f")
                if t2_dt < t1_dt: t2_dt += timedelta(days=1)
                
                # Column indices based on your log_pulsation_data function:
                # 0: Start Time
                # 1: End Time
                # 2: Start Frame
                # 3: End Frame
                # 4: Peak Frame
                # 5: Int (MAD)
                # 6: Int (Raw)
                # 7: Region
                # 8: FWHM (s)
                # 9: FWHM Start
                # 10: FWHM End
                # 11: Source File (last item)

                intervals.append({
                    "t1": t1_dt, "t2": t2_dt, 
                    "start_frame": int(parts[2]), "end_frame": int(parts[3]), 
                    "peak_frame": int(parts[4]), 
                    "intensity_mad": float(parts[5]),
                    "intensity_raw": float(parts[6]),
                    "region": int(parts[7]),
                    "fwhm_sec": float(parts[8]), 
                    "source_file": parts[-1] # Always take the last item as filename
                })
            except (ValueError, IndexError) as e:
                # print(f"Skipping line: {line} -> Error: {e}") # Debugging
                continue

    if not intervals: raise ValueError("Could not parse any interval data.")
    if grid_dims is None: grid_dims = (6, 6)

    # Return pixel_size as the third item in the tuple
    return intervals, grid_dims, pixel_size

def build_frame_map(tiff_files):
    # Goal: Create a map of global frame indices to specific files and their local frame counts.
    # Input: tiff_files (list of str)
    # Output: A tuple containing (frame_map, total_frames).
    #         - frame_map: A list of dictionaries, one for each file.
    #         - total_frames: The total number of frames across all files (int).
    frame_map, total_frames = [], 0
    print("Building TIFF file frame map...")
    for f_path in tiff_files:
        with tifffile.TiffFile(f_path) as tif:
            num_pages = len(tif.pages)
            frame_map.append({'path': f_path, 'start_frame': total_frames, 'count': num_pages})
            total_frames += num_pages
    return frame_map, total_frames

def grouper(iterable, n):
    # Goal: Group data from an iterable into non-overlapping fixed-length chunks.
    # Input: iterable (any iterable), n (int, chunk size)
    # Output: A generator that yields chunks as tuples.
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def calculate_vmin_vmax(tiff_files, files_to_sample=5, frames_per_file=10):
    """
    Calculates vmin/vmax by picking random files from the dataset
    and random frames within those files.
    """
    
    if not tiff_files: return 0, 1000

    # 1. Pick N random files from the list (spreads coverage across the night)
    # If we have fewer files than requested, just use all of them.
    num_files = len(tiff_files)
    if num_files > files_to_sample:
        selected_files = random.sample(tiff_files, files_to_sample)
    else:
        selected_files = tiff_files

    sampled_pixels = []

    try:
        for f_path in selected_files:
            with tifffile.TiffFile(f_path) as tif:
                total_frames = len(tif.pages)
                
                # 2. Pick M random frame indices from this file
                if total_frames > frames_per_file:
                    # Replace=False ensures we don't pick the same frame twice
                    indices = np.random.choice(total_frames, frames_per_file, replace=False)
                    indices.sort() # Sorting sometimes helps I/O speed
                else:
                    indices = range(total_frames)

                # 3. Load ONLY those specific frames
                data_chunk = tif.asarray(key=indices)
                sampled_pixels.append(data_chunk)

        if not sampled_pixels:
            return 0, 1000

        # 4. Flatten all samples into one big array for stats
        full_sample = np.concatenate(sampled_pixels, axis=0)
        
        vmin = np.percentile(full_sample, 5)
        vmax = np.percentile(full_sample, 99.8)
        
        print(f"Estimation complete: vmin={vmin:.2f}, vmax={vmax:.2f}")
        return vmin, vmax

    except Exception as e:
        print(f"Warning: Could not calculate contrast: {e}. Using defaults.")
        return 100, 1500

def extract_text_from_log(folder_path):
    """Finds and reads the .log file in the directory."""
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.log'):
            try:
                with open(os.path.join(folder_path, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading log file: {e}")
    return None

def get_start_end_time(log_text):
    """Parses start/end time from log text."""
    if not log_text: return datetime.now(), datetime.now()
    try:
        t_start = log_text.find("TimeStart1") + 25
        d_start = log_text.find("Date") + 25
        start_str = log_text[d_start:d_start+10] + " " + log_text[t_start:t_start+12]
        return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S.%f"), datetime.now() # End time not strictly needed for player
    except (ValueError, IndexError):
        return datetime.now(), datetime.now()

def save_pulsation_video(output_folder, interval, frame_map, vmin, vmax, grid_dims, 
                         image_size, pixel_size, session_start_time, fps, 
                         pre_roll, post_roll, show_boxes=True, playback_speed=1):
    """
    Saves a specific pulsation interval to an MP4 video file with speed control and detailed titles.
    """
    # 1. Determine Frame Range
    total_dataset_frames = sum(f['count'] for f in frame_map)
    start_frame = max(0, interval['start_frame'] - pre_roll)
    end_frame = min(total_dataset_frames, interval['end_frame'] + post_roll)
    
    # 2. Load the Frames
    print(f"Loading frames {start_frame} to {end_frame} for export...")
    raw_frames = load_frames(start_frame, end_frame, frame_map)
    
    if raw_frames.size == 0:
        print("Error: No frames found for this range.")
        return

    # 3. Apply Speed (Frame Skipping)
    # If speed is 1, take all. If 4, take every 4th frame.
    frames = raw_frames[::playback_speed]

    # 4. Build Filename: V_MMDDYY_HH_MM_SS_MsMs_REGION_SPEEDx.mp4
    video_start_time = session_start_time + timedelta(seconds=start_frame / fps)
    timestamp_str = video_start_time.strftime('%m%d%y_%H_%M_%S_%f')[:-4] 
    filename = f"V_{timestamp_str}_{interval['region']}_{playback_speed}x.mp4"
    save_path = os.path.join(output_folder, filename)

    # 5. Setup Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black') 
    
    im = ax.imshow(frames[0], cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])

    rect_patches = []
    if show_boxes:
        region_coords = generate_grid_regions(grid_dims, image_size, (pixel_size, pixel_size))
        for (y, x) in region_coords:
            rect = patches.Rectangle((x, y), pixel_size, pixel_size, linewidth=2, edgecolor='red', facecolor='none')
            rect_patches.append(rect)
            ax.add_patch(rect)

    def update(i):
        # Calculate the TRUE global frame number considering the speed skip
        current_global = start_frame + (i * playback_speed)
        im.set_data(frames[i])
        
        # --- Update Box Colors ---
        if show_boxes:
            for r_idx, rect in enumerate(rect_patches):
                color = 'red'
                if interval['region'] == r_idx:
                    if interval['start_frame'] <= current_global <= interval['end_frame']:
                        color = 'yellow'
                    if (interval['peak_frame'] - 15) <= current_global <= (interval['peak_frame'] + 15):
                        color = 'lime'
                rect.set_edgecolor(color)

        # --- Update Title with Detailed Info ---
        # A. Find Filename
        current_filename = "Unknown"
        for f_info in frame_map:
            if f_info['start_frame'] <= current_global < (f_info['start_frame'] + f_info['count']):
                current_filename = os.path.basename(f_info['path'])
                break
        
        # B. Calculate Time
        current_time = session_start_time + timedelta(seconds=current_global / fps)
        time_str = current_time.strftime('%H:%M:%S.%f')[:-3]

        # C. Construct Title Parts
        title_parts = [
            f"File: {current_filename}",
            f"Time: {time_str}",
            f"Frame: {current_global}/{total_dataset_frames}", 
            f"Speed: {playback_speed}x"
        ]
        
        ax.set_title(" | ".join(title_parts), color='white', fontsize=9)
        return [im] + rect_patches

    # Use FFMpegWriter
    print(f"Exporting video ({playback_speed}x speed) to: {save_path}")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    
    plt.close(fig) 
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(save_path, writer=writer)
    print("Video saved successfully.")


# --- PLAYER CLASSES ---------------------------------------------------------------------------------------------

class IntervalPlayer:
    def __init__(self, frames, vmin, vmax, global_start_frame, interval_data, grid_dims, image_size, pixel_size, fps):
        # Goal: Initialize an interactive player for a single detected event interval.
        # Input: frames (3D np.ndarray), vmin (float), vmax (float), global_start_frame (int), interval_data (dict), grid_dims (tuple), image_size (tuple), pixel_size (int)
        # Output: None
        self.frames = frames
        self.global_start_frame = global_start_frame
        self.interval_data = interval_data
        self.fps = fps # Store FPS
        self.num_frames = frames.shape[0]
        self.vmin = vmin
        self.vmax = vmax

        self.current_frame_idx = 0
        self.is_paused = True
        self.playback_speeds = [1, 2, 5, 10, 20]
        self.speed_idx = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        self.region_coords = generate_grid_regions(grid_dims, image_size, (pixel_size, pixel_size))
        
        region_idx = self.interval_data['region']
        if region_idx < len(self.region_coords):
            top_left_y, top_left_x = self.region_coords[region_idx]
            self.rect_patch = patches.Rectangle(
                (top_left_x, top_left_y), pixel_size, pixel_size, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            self.ax.add_patch(self.rect_patch)
        else:
            print(f"Warning: Region index {region_idx} out of bounds for grid.")
            self.rect_patch = None

        self.im = self.ax.imshow(self.frames[0], cmap='turbo', vmin=self.vmin, vmax=self.vmax)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        
        self.timer = self.fig.canvas.new_timer(interval=15)
        self.timer.add_callback(self._timer_update)
        self._update_display()

    def _timer_update(self):
        if self.is_paused: return
        speed = self.playback_speeds[self.speed_idx]
        self.current_frame_idx = (self.current_frame_idx + speed) % self.num_frames
        self._update_display()
        self.fig.canvas.draw_idle()
    
    def _update_display(self):
        self.im.set_data(self.frames[self.current_frame_idx])
        self._update_box_color()
        self._update_title()

    def _update_box_color(self):
        if not self.rect_patch: return
        current_global_frame = self.global_start_frame + self.current_frame_idx
        peak_frame = self.interval_data['peak_frame']
        start_frame = self.interval_data['start_frame']
        end_frame = self.interval_data['end_frame']
        
        if (peak_frame - 15) <= current_global_frame <= (peak_frame + 15):
            self.rect_patch.set_edgecolor('lime')
        elif start_frame <= current_global_frame <= end_frame:
            self.rect_patch.set_edgecolor('yellow')
        else:
            self.rect_patch.set_edgecolor('red')

    def _update_title(self):
        speed = self.playback_speeds[self.speed_idx]
        
        time_offset = self.current_frame_idx / self.fps
        
        frame_diff_from_event_start = (self.global_start_frame + self.current_frame_idx) - self.interval_data['start_frame']
        current_time = self.interval_data['t1'] + timedelta(seconds=frame_diff_from_event_start / self.fps)
        
        time_str = current_time.strftime('%H:%M:%S.%f')[:-3]
        filename = os.path.basename(self.interval_data.get('source_file', 'Unknown'))

        title_parts = [
            f"File: {filename}",
            f"Time: {time_str}",
            f"Global Frame: {self.global_start_frame + self.current_frame_idx}",
            f"Speed: {speed}x"
        ]

        if self.is_paused: title_parts.append("[PAUSED]")
        self.ax.set_title(" | ".join(title_parts), fontsize=10)

    def _on_key(self, event):
        if event.key == ' ':
            self.is_paused = not self.is_paused
            if not self.is_paused: self.timer.start()
        elif event.key == 'r': self.current_frame_idx = 0
        elif event.key == 'x': self.speed_idx = (self.speed_idx + 1) % len(self.playback_speeds)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_close(self, event): self.timer.stop()
    def show(self): plt.show()

class FullMoviePlayer:
    def __init__(self, frame_map, total_frames, vmin, vmax, all_intervals, buffer_size, grid_dims, image_size, pixel_size, session_start_time, fps):
        # Goal: Initialize an interactive player for the entire video sequence, highlighting events as they occur.
        # Input: frame_map (list of dict), total_frames (int), vmin (float), vmax (float), all_intervals (list of dict), buffer_size (int), grid_dims (tuple), image_size (tuple), pixel_size (int), start_time (Time), fps (float)
        # Output: None
        self.frame_map, self.total_frames, self.vmin, self.vmax = frame_map, total_frames, vmin, vmax
        self.session_start_time = session_start_time
        self.fps = fps
        self.all_intervals = all_intervals
        self.buffer_size = buffer_size
        self.current_frame_idx, self.is_paused, self.speed_idx = 0, True, 0
        self.playback_speeds = [1, 2, 5, 10, 20, 50, 100]
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_buffer, self.next_buffer_future, self.current_buffer_start_idx = None, None, -1
        
        print("Loading initial frame buffer...")
        self.current_buffer_start_idx = 0
        self.current_buffer = load_frames(0, self.buffer_size, self.frame_map)
        self._preload_next_chunk()

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        self.region_coords = generate_grid_regions(grid_dims, image_size, (pixel_size, pixel_size))
        self.rect_patches = []
        for (y, x) in self.region_coords:
            rect = patches.Rectangle((x, y), pixel_size, pixel_size, linewidth=2, edgecolor='red', facecolor='none', alpha=1)
            self.rect_patches.append(rect)
            self.ax.add_patch(rect)

        self.im = self.ax.imshow(self.current_buffer[0], cmap='turbo', vmin=self.vmin, vmax=self.vmax)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        
        self.timer = self.fig.canvas.new_timer(interval=15)
        self.timer.add_callback(self._timer_update)
        self._update_display()

    def _preload_next_chunk(self):
        next_chunk_start = self.current_buffer_start_idx + self.buffer_size
        if next_chunk_start < self.total_frames:
            print(f"Pre-loading frames from {next_chunk_start}...")
            self.next_buffer_future = self.executor.submit(load_frames, next_chunk_start, next_chunk_start + self.buffer_size, self.frame_map)
        else:
            self.next_buffer_future = None

    def _swap_buffers(self):
        print("Swapping buffers...")
        self.current_buffer = self.next_buffer_future.result()
        self.current_buffer_start_idx += self.buffer_size
        self._preload_next_chunk()

    def _timer_update(self):
        if self.is_paused: return
        self.current_frame_idx += self.playback_speeds[self.speed_idx]
        if self.current_frame_idx >= self.total_frames: self.current_frame_idx = 0
        self._update_display()
        self.fig.canvas.draw_idle()

    def _update_display(self):
        buffer_end_idx = self.current_buffer_start_idx + len(self.current_buffer)
        if self.current_frame_idx >= buffer_end_idx:
            if self.next_buffer_future: self._swap_buffers()
            else: self.current_frame_idx = 0

        if not (self.current_buffer_start_idx <= self.current_frame_idx < self.current_buffer_start_idx + len(self.current_buffer)):
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = ThreadPoolExecutor(max_workers=1)
            print(f"Buffer miss. Force loading frames from {self.current_frame_idx}...")
            self.current_buffer_start_idx = self.current_frame_idx
            self.current_buffer = load_frames(self.current_frame_idx, self.current_frame_idx + self.buffer_size, self.frame_map)
            self._preload_next_chunk()
        
        local_idx = self.current_frame_idx - self.current_buffer_start_idx
        self.im.set_data(self.current_buffer[local_idx])
        self._update_all_box_colors()
        self._update_title()

    def _update_all_box_colors(self):
        active_regions = {i: 'red' for i in range(len(self.rect_patches))}
        
        for interval in self.all_intervals:
            region_idx = interval['region']
            if interval['start_frame'] <= self.current_frame_idx <= interval['end_frame']:
                active_regions[region_idx] = 'yellow'
            if (interval['peak_frame'] - 15) <= self.current_frame_idx <= (interval['peak_frame'] + 15):
                active_regions[region_idx] = 'lime'
        
        for i, rect in enumerate(self.rect_patches):
            rect.set_edgecolor(active_regions[i])


    def _update_title(self):
        # 1. Calculate Time
        current_time = self.session_start_time + timedelta(seconds=self.current_frame_idx / self.fps)
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # 2. Find Filename from frame_map
        current_filename = "Unknown"
        # We iterate to find which file contains the current global frame
        for f_info in self.frame_map:
            if f_info['start_frame'] <= self.current_frame_idx < (f_info['start_frame'] + f_info['count']):
                current_filename = os.path.basename(f_info['path'])
                break

        title_parts = [
            f"File: {current_filename}",
            f"Time: {time_str}",
            f"Frame: {self.current_frame_idx}/{self.total_frames}", 
            f"Speed: {self.playback_speeds[self.speed_idx]}x"
        ]
        
        if self.is_paused: title_parts.append("[PAUSED]")
        self.ax.set_title(" | ".join(title_parts), fontsize=10)

    def _on_key(self, event):
        if event.key in ['left', 'right']: self.is_paused = True; self.timer.stop()
        
        if event.key == ' ':
            self.is_paused = not self.is_paused
            if not self.is_paused: self.timer.start()
        elif event.key == 'r': self.current_frame_idx = 0
        elif event.key == 'x': self.speed_idx = (self.speed_idx + 1) % len(self.playback_speeds)
        elif event.key == 'right': self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 250)
        elif event.key == 'left': self.current_frame_idx = max(0, self.current_frame_idx - 250)
        
        self._update_display()
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def show(self): plt.show()

# --- MAIN EXECUTION ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Configuration ---
    DATA_FOLDER = "Big_Data\\Day_1\\ut02"
    REPORT_FILE = "analysis_output\\Day_1_output\\ut02_report.txt"
    BUFFER_SIZE = 1000      # Frames to hold in memory for the full movie player
    IMAGE_SIZE = (128, 128) # Must match the analysis script
    PIXEL_SIZE = 6          # Must match the analysis script
    FPS = 56.7              # FPS to match Times to Frames

    log_text = extract_text_from_log(DATA_FOLDER)
    session_start_time, _ = get_start_end_time(log_text)
    print(f"Session Start Time determined: {session_start_time}")

    try:
        tiff_files = sorted([os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.tif', '.tiff'))])
        if not tiff_files: raise FileNotFoundError(f"No TIFF files found in '{DATA_FOLDER}'.")

        frame_map, total_frames = build_frame_map(tiff_files)
        vmin, vmax = calculate_vmin_vmax(tiff_files)
        
        all_intervals, grid_dims, PIXEL_SIZE = parse_pulsation_report(REPORT_FILE)
        print(f"Successfully parsed {len(all_intervals)} intervals with a {grid_dims[0]}x{grid_dims[1]} grid.")
        print(f"Detected Pixel Size: {PIXEL_SIZE}x{PIXEL_SIZE}")

        # --- Mode Selection ---
        while True:
            mode = input("\nChoose a player mode:\n"
                         "  1: Full Movie Player\n"
                         "  2: Individual Pulsation Player\n"
                         "  3: Manual Video Generator\n"
                         "Enter choice (1, 2, or 3): ")
            if mode in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")

        # --- Execute Chosen Mode ---
        if mode == '1':
            print("\nStarting Full Movie Player...")
            player = FullMoviePlayer(
                frame_map=frame_map, total_frames=total_frames,
                vmin=vmin, vmax=vmax, all_intervals=all_intervals, 
                buffer_size=BUFFER_SIZE, grid_dims=grid_dims,
                image_size=IMAGE_SIZE, pixel_size=PIXEL_SIZE,
                session_start_time=session_start_time, fps=FPS
            )
            player.show()

        elif mode == '2':
            if not all_intervals:
                print("\nNo intervals to display.")
            else:
                print(f"\nStarting Interval Player for {len(all_intervals)} intervals...")
                BUFFER_FRAMES = 100 # Pre/post-roll frames for interval playback
                for i, interval in enumerate(all_intervals):
                    print("-" * 50)
                    print(f"Preparing interval {i+1}/{len(all_intervals)} (Region {interval['region']})...")
                    display_start = max(0, interval['start_frame'] - BUFFER_FRAMES)
                    display_end = min(total_frames, interval['end_frame'] + BUFFER_FRAMES)
                    
                    print(f"Loading frames {display_start} to {display_end}...")
                    interval_frames = load_frames(display_start, display_end, frame_map)
                    
                    if interval_frames.size > 0:
                        player = IntervalPlayer(
                            frames=interval_frames, vmin=vmin, vmax=vmax,
                            global_start_frame=display_start, interval_data=interval,
                            grid_dims=grid_dims, image_size=IMAGE_SIZE, pixel_size=PIXEL_SIZE, fps = FPS
                        )
                        player.show()
                    else:
                        print("Warning: Could not load frames for this interval. Skipping.")

        elif mode == '3':
            start_frame_input = input("Enter the Start Frame of the pulsation (from report): ").strip()
            if start_frame_input.isdigit():
                start_frame_req = int(start_frame_input)
                target_interval = next((p for p in all_intervals if p['start_frame'] == start_frame_req), None)
                
                if target_interval:
                    pre_roll = int(input("How many frames BEFORE the start frame? "))
                    post_roll = int(input("How many frames AFTER the end frame? "))
                    show_boxes = input("Show boxes in video? (y/n): ").lower() == 'y'
                    
                    # --- NEW: ASK FOR SPEED ---
                    speed_input = input("Enter playback speed (e.g., 1, 2, 4, 10): ").strip()
                    playback_speed = int(speed_input) if speed_input.isdigit() and int(speed_input) > 0 else 1
                    
                    video_out = "Saved_videos" 
                    os.makedirs(video_out, exist_ok=True)
                    
                    save_pulsation_video(
                        video_out, target_interval, frame_map, vmin, vmax, 
                        grid_dims, IMAGE_SIZE, PIXEL_SIZE, session_start_time, FPS,
                        pre_roll, post_roll, show_boxes, playback_speed
                    )
                else:
                    print(f"No pulsation found starting at frame {start_frame_req}.")
            else:
                print("Invalid input.")

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\n--- ERROR ---")
        print(f"{e}")


