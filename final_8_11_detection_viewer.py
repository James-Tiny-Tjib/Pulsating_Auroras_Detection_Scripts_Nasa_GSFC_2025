import os
import re
from datetime import datetime, timedelta
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
import itertools

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
    
    date_str, start_hour_str, intervals = "", "", []
    grid_dims = None
    grid_pattern = re.compile(r'\((\d+)x(\d+)\)')

    for line in lines:
        if line.startswith("Date:"): date_str = line.split("Date:")[1].strip()
        elif line.startswith("UT Hour of Start:"): start_hour_str = line.split("UT Hour of Start:")[1].strip().zfill(2)
        elif "Region Reference Grid" in line:
            match = grid_pattern.search(line)
            if match:
                grid_dims = (int(match.group(1)), int(match.group(2)))

        # Expect 11 columns with the FWHM start/end data
        parts = line.strip().split()
        if len(parts) >= 11 and ":" in parts[0] and "." in parts[0]:
            try:
                t1_dt = datetime.strptime(f"{date_str} {parts[0]}", "%Y-%m-%d %H:%M:%S.%f")
                t2_dt = datetime.strptime(f"{date_str} {parts[1]}", "%Y-%m-%d %H:%M:%S.%f")
                if t2_dt < t1_dt: t2_dt += timedelta(days=1)
                
                intervals.append({
                    "t1": t1_dt, "t2": t2_dt, 
                    "start_frame": int(parts[2]), "end_frame": int(parts[3]), 
                    "peak_frame": int(parts[4]), "region": int(parts[6]),
                    "fwhm_sec": float(parts[7]), "source_file": parts[10]
                })
            except (ValueError, IndexError):
                continue

    if not intervals: raise ValueError("Could not parse any interval data from the report.")
    
    if grid_dims is None:
        print("Warning: Grid dimensions not found in report. Defaulting to (6, 6).")
        grid_dims = (6, 6)

    return intervals, grid_dims

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

def calculate_vmin_vmax_by_chunks(tiff_files, chunk_size=2):
    # Goal: Efficiently calculate image intensity limits (vmin/vmax) for color scaling without loading all files into memory at once.
    # Input: tiff_files (list of str), chunk_size (int)
    # Output: A tuple (vmin, vmax) containing the calculated percentile values (tuple of floats).
    print(f"Calculating vmin/vmax by loading {chunk_size} file(s) at a time...")

    reservoir_size = 100
    sample_frames = []
    frames_seen = 0

    for file_chunk in grouper(tiff_files, chunk_size):
        print(f"  Processing chunk: {[os.path.basename(f) for f in file_chunk]}")
        
        chunk_frame_map, _ = build_frame_map(file_chunk)
        if not chunk_frame_map: continue
        
        chunk_total_frames = sum(f['count'] for f in chunk_frame_map)
        chunk_frames = load_frames(0, chunk_total_frames, chunk_frame_map)

        # Apply Reservoir Sampling to get a representative sample of frames
        for i in range(chunk_frames.shape[0]):
            frame = chunk_frames[i]
            if frames_seen < reservoir_size:
                sample_frames.append(frame)
            else:
                j = np.random.randint(0, frames_seen + 1)
                if j < reservoir_size:
                    sample_frames[j] = frame
            frames_seen += 1

    if not sample_frames:
        print("Error: Could not sample any frames.")
        return 0, 1
    
    vmin = np.percentile(sample_frames, 5)
    vmax = np.percentile(sample_frames, 99.8)

    print(f"\nCalculation complete after seeing {frames_seen} frames.")
    print(f"Global vmin={vmin:.2f}, vmax={vmax:.2f}")
    return vmin, vmax

# --- PLAYER CLASSES ---------------------------------------------------------------------------------------------

class IntervalPlayer:
    def __init__(self, frames, vmin, vmax, global_start_frame, interval_data, grid_dims, image_size, pixel_size):
        # Goal: Initialize an interactive player for a single detected event interval.
        # Input: frames (3D np.ndarray), vmin (float), vmax (float), global_start_frame (int), interval_data (dict), grid_dims (tuple), image_size (tuple), pixel_size (int)
        # Output: None
        self.frames = frames
        self.global_start_frame = global_start_frame
        self.interval_data = interval_data
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
        t1 = self.interval_data['t1'].strftime('%H:%M:%S')
        t2 = self.interval_data['t2'].strftime('%H:%M:%S')
        fwhm_str = f" | FWHM: {self.interval_data['fwhm_sec']:.2f}s"
        info_str = f"{t1} to {t2} (Region {self.interval_data['region']}){fwhm_str}"
        
        title_parts = [
            f"Event: {info_str}",
            f"Local Frame: {self.current_frame_idx}/{self.num_frames}\n",
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
    def __init__(self, frame_map, total_frames, vmin, vmax, all_intervals, buffer_size, grid_dims, image_size, pixel_size):
        # Goal: Initialize an interactive player for the entire video sequence, highlighting events as they occur.
        # Input: frame_map (list of dict), total_frames (int), vmin (float), vmax (float), all_intervals (list of dict), buffer_size (int), grid_dims (tuple), image_size (tuple), pixel_size (int)
        # Output: None
        self.frame_map, self.total_frames, self.vmin, self.vmax = frame_map, total_frames, vmin, vmax
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
            rect = patches.Rectangle((x, y), pixel_size, pixel_size, linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
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
        title_parts = [f"Global Frame: {self.current_frame_idx}/{self.total_frames}", f"Speed: {self.playback_speeds[self.speed_idx]}x"]
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
    DATA_FOLDER = "Big_Data\\Day_1\\ut15"
    REPORT_FILE = "pulsation_report_grid.txt"
    BUFFER_SIZE = 1000      # Frames to hold in memory for the full movie player
    IMAGE_SIZE = (128, 128) # Must match the analysis script
    PIXEL_SIZE = 6          # Must match the analysis script

    try:
        tiff_files = sorted([os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.tif', '.tiff'))])
        if not tiff_files: raise FileNotFoundError(f"No TIFF files found in '{DATA_FOLDER}'.")

        frame_map, total_frames = build_frame_map(tiff_files)
        vmin, vmax = calculate_vmin_vmax_by_chunks(tiff_files, chunk_size=2)
        
        all_intervals, grid_dims = parse_pulsation_report(REPORT_FILE)
        print(f"Successfully parsed {len(all_intervals)} intervals with a {grid_dims[0]}x{grid_dims[1]} grid.")

        # --- Mode Selection ---
        while True:
            mode = input("\nChoose a player mode:\n"
                         "  1: Full Movie Player\n"
                         "  2: Individual Pulsation Player\n"
                         "Enter choice (1 or 2): ")
            if mode in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")

        # --- Execute Chosen Mode ---
        if mode == '1':
            print("\nStarting Full Movie Player...")
            player = FullMoviePlayer(
                frame_map=frame_map, total_frames=total_frames,
                vmin=vmin, vmax=vmax, all_intervals=all_intervals, 
                buffer_size=BUFFER_SIZE, grid_dims=grid_dims,
                image_size=IMAGE_SIZE, pixel_size=PIXEL_SIZE
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
                            grid_dims=grid_dims, image_size=IMAGE_SIZE, pixel_size=PIXEL_SIZE
                        )
                        player.show()
                    else:
                        print("Warning: Could not load frames for this interval. Skipping.")

        print("-" * 50)
        print("Player closed.")

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\n--- ERROR ---")
        print(f"{e}")