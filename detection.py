"""
Title: Pulsating Aurora Auto-Detection Module
Author: James Hui
Affiliation: River Hill High School, UMD
Email: james.y.hui@gmail.com
GitHub: https://github.com/James-Tiny-Tjib
Date Created: 2025-01-06
Last Updated: 2025-01-06
"""

import tifffile
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.dates as mdates
from scipy.signal import savgol_filter, find_peaks
from functools import partial
import multiprocessing

# --- STAGE 1: DATA EXTRACTION --------------------------------------------------------------------------------------

def load_folders(folder_path):
    # Goal: Finds all TIFF files in a directory and sorts them by name.
    # Input: folder_path (str)
    # Output: A sorted list of full file paths (list of str).
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return []

    file_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(folder_path, f))
    ])

    if not file_paths:
        print(f"No TIFF files found in '{folder_path}'")
    return file_paths

def generate_grid_regions(grid_dims, image_size, box_size):
    # Goal: Generate the top-left coordinates for an evenly spaced grid of regions.
    # Input: grid_dims (tuple), image_size (tuple), box_size (tuple)
    # Output: A list of (y, x) tuples for the top-left corner of each grid region (list of tuples).

    import matplotlib
    matplotlib.use('Agg')
    
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

# def get_grid_averages(filepath, pixel_size, regions, sigma_clip_threshold, MAD=True):
#     # Goal: Calculate the average intensity time series for all regions within a single TIFF file, with optional outlier rejection.
#     # Input: filepath (str), pixel_size (int), regions (list of tuples), sigma_clip_threshold (int), MAD (bool)
#     # Output: A tuple containing (grid_averages, filepath).
#     #         - grid_averages: A list of 1D NumPy arrays. The list has `len(regions)` elements. Each array contains the intensity time series for one region within the file.
#     #         - filepath: The original filepath (str).
#     try:
#         image_stack = tifffile.imread(filepath)
#         num_frames = image_stack.shape[0]
#         num_regions = len(regions)
        
#         # 1. Create TWO arrays instead of one
#         grid_averages_mad = [np.zeros(num_frames) for _ in range(num_regions)]
#         grid_averages_raw = [np.zeros(num_frames) for _ in range(num_regions)] # <--- NEW

#         if MAD:
#             for i, (r, c) in enumerate(regions):
#                 region_stack = image_stack[:, r:r+pixel_size, c:c+pixel_size]
                
#                 for frame_idx in range(num_frames):
#                     frame_data = region_stack[frame_idx, :, :]
                    
#                     # 2. ALWAYS calculate Raw Mean first
#                     grid_averages_raw[i][frame_idx] = np.mean(frame_data) # <--- NEW
                    
#                     # --- ROBUST OUTLIER REJECTION LOGIC (For MAD Array) ---
#                     median_val = np.median(frame_data)
#                     mad = np.median(np.abs(frame_data - median_val))
                    
#                     if mad == 0:
#                         grid_averages_mad[i][frame_idx] = median_val
#                         continue

#                     threshold = median_val + sigma_clip_threshold * mad * 1.4826
#                     valid_pixels = frame_data[frame_data < threshold]
                    
#                     if valid_pixels.size > 0:
#                         grid_averages_mad[i][frame_idx] = np.mean(valid_pixels)
#                     else:
#                         grid_averages_mad[i][frame_idx] = median_val 
#         else:
#             for i, (r, c) in enumerate(regions):
#                 subset = image_stack[:, r:r+pixel_size, c:c+pixel_size]
#                 mean_val = np.mean(subset, axis=(1, 2))
#                 grid_averages_mad[i] = mean_val
#                 grid_averages_raw[i] = mean_val # If MAD is off, they are the same

#         # 3. Return BOTH arrays
#         return grid_averages_mad, grid_averages_raw, filepath 
#     except Exception as e:
#         print(f"Error processing {os.path.basename(filepath)}: {e}")
#         return None

def get_grid_averages(filepath, pixel_size, regions, sigma_clip_threshold, MAD=True):
    try:
        # Load the entire stack (T, H, W)
        image_stack = tifffile.imread(filepath)
        num_frames = image_stack.shape[0]
        num_regions = len(regions)
        
        # Pre-allocate output lists
        grid_averages_mad = []
        grid_averages_raw = []

        if MAD:
            # Vectorized Processing per Region
            for r, c in regions:
                # 1. Extract the entire time-cube for this region (T, pixel_size, pixel_size)
                # This creates a view, instant and low memory
                region_stack = image_stack[:, r:r+pixel_size, c:c+pixel_size]
                
                # 2. Calculate Raw Mean (Vectorized across axes 1 and 2)
                # Result shape: (num_frames,)
                raw_means = np.mean(region_stack, axis=(1, 2))
                grid_averages_raw.append(raw_means)

                # 3. Vectorized MAD Filter
                # Calculate Median of each frame
                medians = np.median(region_stack, axis=(1, 2)) # Shape: (T,)
                
                # Calculate Absolute Deviation from the median (Broadcasting subtraction)
                # region_stack is (T, H, W), medians is (T,). We need (T, 1, 1) for broadcasting
                abs_diff = np.abs(region_stack - medians[:, None, None])
                
                # Calculate MAD for each frame
                mads = np.median(abs_diff, axis=(1, 2))
                
                # Calculate Thresholds
                thresholds = medians + (sigma_clip_threshold * mads * 1.4826)
                
                # 4. Create a Mask for valid pixels
                # Broadcast thresholds to (T, 1, 1) to compare against full stack
                mask = region_stack < thresholds[:, None, None]
                
                # 5. Apply Filter using np.where and np.nanmean
                # Where the mask is False (outlier), replace with NaN
                # We cast to float32 to allow NaNs (integers can't hold NaN)
                masked_stack = np.where(mask, region_stack, np.nan)
                
                # Calculate mean ignoring NaNs
                # This gives us the mean of only the "valid" pixels for every frame at once
                clean_means = np.nanmean(masked_stack, axis=(1, 2))
                
                # 6. Handle the edge case where MAD == 0 (flat image)
                # If MAD is 0, the math above might fail or act weird. 
                # Revert those specific frames to the simple median.
                clean_means = np.where(mads == 0, medians, clean_means)
                
                # 7. Handle the edge case where ALL pixels were clipped (result is NaN)
                # Revert those to median
                clean_means = np.where(np.isnan(clean_means), medians, clean_means)
                
                grid_averages_mad.append(clean_means)

        else:
            # Extremely fast simple averaging
            for r, c in regions:
                subset = image_stack[:, r:r+pixel_size, c:c+pixel_size]
                mean_val = np.mean(subset, axis=(1, 2))
                grid_averages_mad.append(mean_val)
                grid_averages_raw.append(mean_val)

        return grid_averages_mad, grid_averages_raw, filepath

    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None

def run_parallel_for_grid(folder_path, pixel_size, grid_dims, image_size, sigma_clip_threshold):
    # Goal: Orchestrate the parallel extraction of intensity data from all TIFF files for all defined grid regions.
    # Input: folder_path (str), pixel_size (int), grid_dims (tuple), image_size (tuple), sigma_clip_threshold (int)
    # Output: A tuple containing (final_arrays, frame_to_file_map, num_regions).
    #         - final_arrays: A list of 1D NumPy arrays. The list has `num_regions` elements. Each array is the concatenated intensity time series for a single region across all files.
    #         - frame_to_file_map: A list where the index is the global frame number and the value is the source filepath (str).
    #         - num_regions: The total number of grid regions analyzed (int).
    file_paths = load_folders(folder_path)
    if not file_paths: 
        return None, None, 0

    regions = generate_grid_regions(grid_dims, image_size, (pixel_size, pixel_size))
    num_regions = len(regions)
    print(f"Analyzing a {grid_dims[0]}x{grid_dims[1]} grid ({num_regions} regions).")

    print(f"\nExtracting grid data from {len(file_paths)} files in parallel...")
    worker_func = partial(get_grid_averages, pixel_size=pixel_size, regions=regions, sigma_clip_threshold=sigma_clip_threshold, MAD=True)    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, file_paths))
    
    valid_results = [r for r in results if r is not None]
    if not valid_results: return None, None, 0

    final_grid_timeseries_mad = [[] for _ in range(num_regions)]
    final_grid_timeseries_raw = [[] for _ in range(num_regions)]
    frame_to_file_map = []

    for grid_data_mad, grid_data_raw, filepath in valid_results: # <--- Unpack 3 items
        if grid_data_mad and len(grid_data_mad) > 0:
            num_frames_in_file = grid_data_mad[0].shape[0]
            for i in range(num_regions):
                final_grid_timeseries_mad[i].append(grid_data_mad[i])
                final_grid_timeseries_raw[i].append(grid_data_raw[i]) # <--- Append Raw
            frame_to_file_map.extend([filepath] * num_frames_in_file)
    
    if not any(final_grid_timeseries_mad):
        print("No valid data extracted.")
        return None, None, None, 0 # Return extra None

    # 2. Concatenate both sets
    final_arrays_mad = [np.concatenate(region_data) for region_data in final_grid_timeseries_mad]
    final_arrays_raw = [np.concatenate(region_data) for region_data in final_grid_timeseries_raw] # <--- NEW
    
    print("Grid data extraction complete.")
    # 3. Return BOTH sets of arrays
    return final_arrays_mad, final_arrays_raw, frame_to_file_map, num_regions

# --- STAGE 2: PEAK DETECTION AND VETTING ----------------------------------------------------------------------------------------------

def calculate_fwhm(raw_data, smoothed_data, peak_idx, prominence):
    # Goal: Calculate the Full Width at Half Maximum (FWHM) of a peak by interpolating on the raw data.
    # Input: raw_data (np.array), smoothed_data (np.array), peak_idx (int), prominence (float)
    # Output: A tuple (fwhm_frames, left_idx, right_idx) representing the width and boundary indices. Returns (-1, -1, -1) on failure.
    peak_value_smoothed = smoothed_data[peak_idx]
    half_height = peak_value_smoothed - (prominence / 2)

    # --- Find left intersection on RAW data ---
    left_idx_interpolated = -1
    for i in range(peak_idx, 0, -1):
        if raw_data[i] <= half_height <= raw_data[i+1]:
            y1, y2 = raw_data[i], raw_data[i+1]
            x1, x2 = i, i+1
            if y2 - y1 == 0: continue
            # Interpolate to find the fractional index
            left_idx_interpolated = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            break
    
    if left_idx_interpolated == -1: return -1, -1, -1

    # --- Find right intersection on RAW data ---
    right_idx_interpolated = -1
    for i in range(peak_idx, len(raw_data) - 1):
        if raw_data[i-1] >= half_height >= raw_data[i]:
            y1, y2 = raw_data[i-1], raw_data[i]
            x1, x2 = i-1, i
            if y2 - y1 == 0: continue
            # Interpolate to find the fractional index
            right_idx_interpolated = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            break

    if right_idx_interpolated == -1: return -1, -1, -1

    return right_idx_interpolated - left_idx_interpolated, left_idx_interpolated, right_idx_interpolated

# def detect_peaks_in_grid(grid_timeseries, params, frame_to_file_map):
#     # Goal: Detect and filter peaks in each region's time series using smoothing and dynamic prominence criteria.
#     # Input: grid_timeseries (list of 1D np.arrays), params (tuple), frame_to_file_map (list of str)
#     # Output: A list of dictionaries, where each dictionary contains detailed properties of a valid detected peak (list of dict).
#     all_detected_peaks = []
    
#     normal_prominence, reduced_prominence, high_base_threshold, fps, find_max, max_width_sec, min_prom_ratio = params

#     initial_peak_count = 0
#     rejected_by_low_prom = 0
#     rejected_by_ratio = 0

#     for i, intensity_data in enumerate(grid_timeseries):
#         smoothing_window = 51
#         if smoothing_window >= len(intensity_data):
#             smoothing_window = len(intensity_data) - 1
#         if smoothing_window % 2 == 0: smoothing_window -=1
#         if smoothing_window < 3: continue

#         smoothed_data = savgol_filter(intensity_data, smoothing_window, 3)
        
#         min_width_samples = int(0.2 * fps) 
#         max_width_samples = int(max_width_sec * fps)

#         # --- DYNAMIC PROMINENCE LOGIC ---
#         # 1. Find all peaks that meet the lower, 'reduced_prominence' threshold.
#         peaks, props = find_peaks(
#             smoothed_data, 
#             prominence=reduced_prominence, 
#             width=(min_width_samples, max_width_samples),
#             wlen=600
#         )
#         initial_peak_count += len(peaks)
        
#         for j, peak_idx_savgol in enumerate(peaks):
#             # 2. Check the base intensity of each potential peak.
#             base_level = (smoothed_data[props["left_bases"][j]] + smoothed_data[props["right_bases"][j]]) / 2
#             actual_prominence = props["prominences"][j]

#             fwhm_frames, left_fwhm_idx, right_fwhm_idx = calculate_fwhm(
#                 intensity_data, smoothed_data, peak_idx_savgol, actual_prominence
#             )
#             fwhm_sec = fwhm_frames / fps if fwhm_frames > 0 else 0.0

#             # 3. If the base is not on a bright background, it must meet the higher 'normal_prominence'.
#             if base_level < high_base_threshold and actual_prominence < normal_prominence:
#                 rejected_by_low_prom += 1
#                 continue
            
#             start_frame = props["left_bases"][j]
#             end_frame = props["right_bases"][j]
#             width = end_frame - start_frame
#             if width > 0 and (actual_prominence / width) < min_prom_ratio:
#                 rejected_by_ratio += 1
#                 continue
            
#             peak_idx = peak_idx_savgol
#             if find_max:
#                 search_slice = intensity_data[start_frame : end_frame + 1]
#                 if search_slice.size > 0:
#                     local_max_idx = np.argmax(search_slice)
#                     peak_idx = start_frame + local_max_idx
            
#             final_intensity_at_peak = intensity_data[peak_idx]
#             source_file = frame_to_file_map[peak_idx]

#             all_detected_peaks.append({
#                 "index": peak_idx,
#                 "intensity": final_intensity_at_peak,
#                 "prominence": actual_prominence,
#                 "start_index": start_frame,
#                 "end_index": end_frame,
#                 "region": i,
#                 "source_file": source_file,
#                 "fwhm_sec": fwhm_sec,
#                 "left_fwhm_idx": left_fwhm_idx,
#                 "right_fwhm_idx": right_fwhm_idx
#             })
    
#     print(f"Initial candidates found: {initial_peak_count}")
#     print(f"Rejected by dynamic prominence filter: {rejected_by_low_prom}")
#     print(f"Rejected by sharpness filter: {rejected_by_ratio}")
#     print(f"Final valid peaks found: {len(all_detected_peaks)}")
#     return all_detected_peaks

def detect_peaks_for_region(args):
    """
    Worker function to find peaks in a single region's time series data.
    This function is called in parallel by the main peak detector.
    """
    # Unpack arguments
    region_idx, intensity_data_mad, intensity_data_raw, params, frame_to_file_map = args    
    # Unpack parameters from the config tuple
    normal_prominence, reduced_prominence, high_base_threshold, fps, find_max, max_width_sec, min_prom_ratio = params

    all_peaks_for_region = []
    rejected_counts = {'low_prom': 0, 'ratio': 0}

    # Apply smoothing filter
    smoothing_window = 51
    if smoothing_window >= len(intensity_data_mad):
        smoothing_window = len(intensity_data_mad) - 1
    if smoothing_window % 2 == 0: smoothing_window -= 1
    if smoothing_window < 3:
        return all_peaks_for_region, rejected_counts

    smoothed_data = savgol_filter(intensity_data_mad, smoothing_window, 3)
    
    min_width_samples = int(0.2 * fps) 
    max_width_samples = int(max_width_sec * fps)

    peaks, props = find_peaks(
        smoothed_data, 
        prominence=reduced_prominence, 
        width=(min_width_samples, max_width_samples),
        wlen=600
    )

    for j, peak_idx_savgol in enumerate(peaks):
        base_level = (smoothed_data[props["left_bases"][j]] + smoothed_data[props["right_bases"][j]]) / 2
        actual_prominence = props["prominences"][j]

        # **FIX**: Removed 'analysis_module.' prefix. 
        # Functions in the same file call each other directly.
        fwhm_frames, left_fwhm_idx, right_fwhm_idx = calculate_fwhm(
            intensity_data_mad, smoothed_data, peak_idx_savgol, actual_prominence
        )
        fwhm_sec = fwhm_frames / fps if fwhm_frames > 0 else 0.0

        if base_level < high_base_threshold and actual_prominence < normal_prominence:
            rejected_counts['low_prom'] += 1
            continue
        
        start_frame = props["left_bases"][j]
        end_frame = props["right_bases"][j]
        width = end_frame - start_frame
        if width > 0 and (actual_prominence / width) < min_prom_ratio:
            rejected_counts['ratio'] += 1
            continue
        
        peak_idx = peak_idx_savgol
        if find_max:
            search_slice = intensity_data_mad[start_frame : end_frame + 1]
            if search_slice.size > 0:
                local_max_idx = np.argmax(search_slice)
                peak_idx = start_frame + local_max_idx
        
        val_mad = intensity_data_mad[peak_idx]
        val_raw = intensity_data_raw[peak_idx]
        source_file = frame_to_file_map[peak_idx]

        all_peaks_for_region.append({
            "index": peak_idx, 
            "intensity_mad": val_mad,
            "intensity_raw": val_raw,
            "intensity": val_mad,
            "prominence": actual_prominence, 
            "start_index": start_frame,
            "end_index": end_frame, 
            "region": region_idx,
            "source_file": source_file, 
            "fwhm_sec": fwhm_sec,
            "left_fwhm_idx": left_fwhm_idx, 
            "right_fwhm_idx": right_fwhm_idx
        })

    return all_peaks_for_region, rejected_counts

def detect_peaks_in_grid(grid_timeseries_mad, grid_timeseries_raw, params, frame_to_file_map):
    # Goal: Detect peaks in each grid region's time series using parallel processing.
    # Input: grid_timeseries (list of np.array), params (tuple), frame_to_file_map (list of str)
    # Output: A list of dictionaries for all valid peaks found across all regions (list of dict).
    all_detected_peaks = []
    
    total_initial_peaks = 0
    total_rejected_low_prom = 0
    total_rejected_ratio = 0
    
    print("\nDetecting peaks in each region in parallel...")
    # Prepare a list of argument tuples for each worker process
    tasks = [
        (i, grid_timeseries_mad[i], grid_timeseries_raw[i], params, frame_to_file_map) 
        for i in range(len(grid_timeseries_mad))
    ]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = executor.map(detect_peaks_for_region, tasks)

    for region_peaks, rejected_counts in results:
        all_detected_peaks.extend(region_peaks)
        total_initial_peaks += len(region_peaks) + rejected_counts['low_prom'] + rejected_counts['ratio']
        total_rejected_low_prom += rejected_counts['low_prom']
        total_rejected_ratio += rejected_counts['ratio']
    
    print("Peak detection complete.")
    print(f"Initial candidates found: {total_initial_peaks}")
    print(f"Rejected by dynamic prominence filter: {total_rejected_low_prom}")
    print(f"Rejected by sharpness filter: {total_rejected_ratio}")
    print(f"Final valid peaks found: {len(all_detected_peaks)}")
    return all_detected_peaks

def deduplicate_peaks(all_peaks, fps, grid_dims):
    # Goal: Group and merge peaks that likely correspond to the same physical event based on time and spatial adjacency.
    # Input: all_peaks (list of dict), fps (float), grid_dims (tuple)
    # Output: A filtered list of dictionaries, keeping only the most intense peak from each group (list of dict).
    if not all_peaks:
        return []

    grid_cols = grid_dims[1]
    all_peaks.sort(key=lambda p: p["index"])
    
    final_peaks = []
    processed_indices = set()
    
    for i, peak1 in enumerate(all_peaks):
        if i in processed_indices:
            continue
            
        broad_time_window = int(0.5 * fps)
        adj_region_time_window = int(2.0 * fps)
        
        group = [peak1]
        
        for j, peak2 in enumerate(all_peaks[i+1:], start=i+1):
            if j in processed_indices:
                continue

            time_diff = peak2["index"] - peak1["index"]
            
            if time_diff >= adj_region_time_window:
                break

            # --- COMBINED GROUPING LOGIC ---
            is_grouped = False
            # Condition 1: Original broad time-based rule (applies to any region).
            if time_diff < broad_time_window:
                is_grouped = True
            # Condition 2: New adjacent-region rule.
            else:
                r1 = peak1["region"]
                r2 = peak2["region"]
                
                if r1 != r2:
                    is_adjacent = False
                    # Case A: r1 is on the left edge.
                    if r1 % grid_cols == 0:
                        if r2 in [r1 + 1, r1 - grid_cols, r1 + grid_cols, r1 - grid_cols + 1, r1 + grid_cols + 1]:
                            is_adjacent = True
                    # Case B: r1 is on the right edge.
                    elif r1 % grid_cols == grid_cols - 1:
                        if r2 in [r1 - 1, r1 - grid_cols, r1 + grid_cols, r1 - grid_cols - 1, r1 + grid_cols - 1]:
                            is_adjacent = True
                    # Case C: r1 is in the middle.
                    else:
                        if r2 in [r1 - 1, r1 + 1, r1 - grid_cols, r1 + grid_cols, 
                                  r1 - grid_cols - 1, r1 - grid_cols + 1, 
                                  r1 + grid_cols - 1, r1 + grid_cols + 1]:
                            is_adjacent = True
                    
                    if is_adjacent:
                        is_grouped = True

            if is_grouped:
                group.append(peak2)
                processed_indices.add(j)

        best_peak = max(group, key=lambda p: p["intensity"])
        final_peaks.append(best_peak)
        processed_indices.add(i)
            
    print(f"After de-duplication, {len(final_peaks)} unique pulsations remain.")
    return final_peaks

# --- STAGE 3: REPORTING -----------------------------------------------------------------------------------------------------------

def extract_text_from_log(folder_path):
    # Goal: Find and read the content of a .log file in a specified directory.
    # Input: folder_path (str)
    # Output: The content of the log file as a string (str), or None if not found/readable.
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.log'):
            try:
                with open(os.path.join(folder_path, fname), 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading log file: {e}")
    return None

def get_start_end_time(log_text):
    # Goal: Parse the experiment's start time from the log file text.
    # Input: log_text (str)
    # Output: A datetime object representing the start time (datetime).
    try:
        t_start = log_text.find("TimeStart1") + 25
        d_start = log_text.find("Date") + 25
        start_str = log_text[d_start:d_start+10] + " " + log_text[t_start:t_start+12]
        return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S.%f")
    except (ValueError, IndexError):
        print("Warning: Could not parse start time from log. Using current time.")
        return datetime.now()

def log_pulsation_data(final_peaks, start_time, fps, prominence, pixel_size, grid_dims):
    # Goal: Generate and write a formatted text report of the final detected pulsations.
    # Input: final_peaks (list of dict), start_time (datetime), fps (float), prominence (int), pixel_size (int), grid_dims (tuple)
    # Output: None. Writes a file named "pulsation_report_grid.txt".

    output_lines = []
    
    output_lines.append(f"Pulsation Detection Report (Method: Savitzky-Golay Filter)")
    output_lines.append(f"Date: {start_time.strftime('%Y-%m-%d')}")
    output_lines.append(f"UT Hour of Start: {start_time.strftime('%H')}")
    output_lines.append("-" * 40)
    output_lines.append(f"Prominence Threshold: {prominence}")
    output_lines.append(f"Pixel Grid Size: {pixel_size}x{pixel_size}")
    
    output_lines.append(f"\nRegion Reference Grid ({grid_dims[0]}x{grid_dims[1]}):")
    region_num = 0
    for r in range(grid_dims[0]):
        line = "  "
        for c in range(grid_dims[1]):
            line += f"{region_num:<3} "
            region_num += 1
        output_lines.append(line)

    output_lines.append("-" * 40)
    
    # 1. Define fixed column widths in the header
    header = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<12} {:<12} {:<10} {:<12} {:<14} {:<14} {:<25}".format(
        "Start Time (t1)", "End Time (t2)", "Start Frame", "End Frame", "Peak Frame", 
        "Int (MAD)", "Int (Raw)", "Region", "FWHM (s)", "FWHM Start", "FWHM End", "Source File"
    )
    output_lines.append(header)
    output_lines.append("=" * len(header))

    for peak in final_peaks:
        peak_frame = peak["index"]
        start_frame = peak["start_index"]
        end_frame = peak["end_index"]

        t1 = start_time + timedelta(seconds=start_frame / fps)
        t2 = start_time + timedelta(seconds=end_frame / fps)
        fwhm_start_frame = peak.get("left_fwhm_idx", -1)
        fwhm_end_frame = peak.get("right_fwhm_idx", -1)
        source_filename = os.path.basename(peak["source_file"])
        
        fwhm_start_str = f"{fwhm_start_frame:.0f}" if fwhm_start_frame != -1 else "N/A"
        fwhm_end_str = f"{fwhm_end_frame:.0f}" if fwhm_end_frame != -1 else "N/A"
        
        # 2. Apply correct formatting to the data line
        line = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<12.2f} {:<12.2f} {:<10} {:<12.2f} {:<14} {:<14} {:<25}".format(
            t1.strftime('%H:%M:%S.%f')[:-4],
            t2.strftime('%H:%M:%S.%f')[:-4],
            start_frame,
            end_frame,
            peak_frame,
            peak["intensity_mad"], # Int (MAD) -> rounded .2f
            peak["intensity_raw"], # Int (Raw) -> rounded .2f
            peak["region"],        # Region    -> No .2f (prints as integer)
            peak["fwhm_sec"],      # FWHM      -> rounded .2f
            fwhm_start_str,
            fwhm_end_str,
            source_filename        # Source File -> Included at the end
        )
        output_lines.append(line)
        
    output_filename = "pulsation_report_grid.txt"
    print(f"\nWriting final report to {output_filename}...")
    with open(output_filename, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    print("Report generation complete.")

def generate_full_timeseries_plots(final_peaks, grid_timeseries, start_time, fps, output_folder, frame_to_file_map, grid_dims):
    # Goal: Generate and save a plot of the complete intensity time series for each grid region, marking detected peaks.
    # Input: final_peaks (list of dict), grid_timeseries (list of np.array), start_time (datetime), fps (float), output_folder (str), frame_to_file_map (list of str), grid_dims (tuple)
    # Output: None. Saves PNG plot files to a subfolder named "full_timeseries_plots".
    if not grid_timeseries or not any(ts.any() for ts in grid_timeseries):
        print("No timeseries data to plot.")
        return

    num_regions = grid_dims[0] * grid_dims[1]

    if not frame_to_file_map:
        print("Warning: Cannot determine TIFF ID for filenames. Using a generic name.")
        tiff_id = "UNKNOWN"
    else:
        first_file = os.path.basename(frame_to_file_map[0])
        tiff_id = first_file[3:10]

    full_plots_folder = os.path.join(output_folder, "full_timeseries_plots")
    os.makedirs(full_plots_folder, exist_ok=True)
    
    print(f"\nGenerating {num_regions} full time-intensity plots...")

    peaks_by_region = {i: [] for i in range(num_regions)}
    for peak in final_peaks:
        region_idx = peak.get("region")
        if region_idx is not None and region_idx in peaks_by_region:
            peaks_by_region[region_idx].append(peak)

    for i in range(num_regions):
        intensity_data = grid_timeseries[i]
        
        total_frames = len(intensity_data)
        time_objects = [start_time + timedelta(seconds=j / fps) for j in range(total_frames)]

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_objects, intensity_data, label=f'Region {i} Full Timeseries', color='dodgerblue', linewidth=0.8)

        region_peaks = peaks_by_region.get(i, [])
        if region_peaks:
            peak_indices = [p['index'] for p in region_peaks]
            valid_peak_indices = [idx for idx in peak_indices if idx < len(intensity_data)]
            peak_times = [start_time + timedelta(seconds=idx / fps) for idx in valid_peak_indices]
            peak_intensities = [intensity_data[idx] for idx in valid_peak_indices]
            ax.plot(peak_times, peak_intensities, 'x', color='red', markersize=8, mew=1.5, label=f'Detected Peaks ({len(peak_times)})')

        ax.set_title(f"Full Time-Intensity Profile - Region {i}")
        ax.set_xlabel("Time (UT)")
        ax.set_ylabel("Average Intensity")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        fig.tight_layout()

        filename = f"F_{tiff_id}_region_{i}.png"
        save_path = os.path.join(full_plots_folder, filename)
        fig.savefig(save_path)
        plt.close(fig)

    print(f"Full timeseries plots saved to '{full_plots_folder}'.")

# def generate_pulsation_plots(final_peaks, grid_timeseries, start_time, fps, output_folder):
#     # Goal: Generate and save a detailed plot for each individual detected pulsation event.
#     # Input: final_peaks (list of dict), grid_timeseries (list of np.array), start_time (datetime), fps (float), output_folder (str)
#     # Output: None. Saves PNG plot files to the specified output folder.
#     if not final_peaks:
#         return
        
#     print(f"\nGenerating {len(final_peaks)} individual pulsation plots...")
#     os.makedirs(output_folder, exist_ok=True)
    
#     frame_buffer = 50

#     for peak in final_peaks:
#         region_idx = peak["region"]
#         intensity_data = grid_timeseries[region_idx]
        
#         start_frame = peak["start_index"]
#         end_frame = peak["end_index"]

#         plot_start_index = max(0, start_frame - frame_buffer)
#         plot_end_index = min(len(intensity_data), end_frame + frame_buffer)
        
#         plot_data = intensity_data[plot_start_index:plot_end_index]
#         plot_time_objects = [start_time + timedelta(seconds=i/fps) for i in range(plot_start_index, plot_end_index)]
        
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(plot_time_objects, plot_data, label=f'Region {region_idx} Intensity', zorder=1.5, linewidth=2 )
        
#         plot_smoothing_window = 51
#         if plot_smoothing_window >= len(plot_data):
#             plot_smoothing_window = len(plot_data) - 1
#         if plot_smoothing_window % 2 == 0: plot_smoothing_window -= 1
#         if plot_smoothing_window > 2:
#             smoothed_plot_data = savgol_filter(plot_data, plot_smoothing_window, 3)
#             ax.plot(plot_time_objects, smoothed_plot_data, label='Smoothed Data', color='orange', linestyle=':', zorder=2)

#         peak_time = start_time + timedelta(seconds=peak["index"] / fps)
#         peak_intensity_raw = peak["intensity"] 
#         ax.plot(peak_time, peak_intensity_raw, 'x', color='red', markersize=10, mew=2, label=f'Peak', zorder=3)
        
#         start_peak_time = start_time + timedelta(seconds=start_frame / fps)
#         end_peak_time = start_time + timedelta(seconds=end_frame / fps)
        
#         ax.axvline(x=start_peak_time, color='green', linestyle='--', label='Peak Start/End', zorder=1)
#         ax.axvline(x=end_peak_time, color='green', linestyle='--', zorder=1)
        
#         # --- FWHM Visualization ---
#         if peak.get("fwhm_sec", 0) > 0 and peak.get("left_fwhm_idx", -1) != -1:
#             peak_idx_local = peak["index"] - plot_start_index
            
#             if 'smoothed_plot_data' in locals() and 0 <= peak_idx_local < len(smoothed_plot_data):
#                 peak_value_smoothed = smoothed_plot_data[peak_idx_local]
#                 half_height = peak_value_smoothed - (peak["prominence"] / 2)

#                 left_fwhm_time = start_time + timedelta(seconds=peak["left_fwhm_idx"] / fps)
#                 right_fwhm_time = start_time + timedelta(seconds=peak["right_fwhm_idx"] / fps)

#                 ax.hlines(y=half_height, xmin=left_fwhm_time, xmax=right_fwhm_time,
#                           color='purple', linestyle='--', label=f'FWHM ({peak["fwhm_sec"]:.2f}s)')
#                 ax.plot([left_fwhm_time, right_fwhm_time], [half_height, half_height], 'o', color='purple')

#         ax.set_title(f"Pulsation Detected in Region {region_idx}")
#         ax.set_xlabel("Time (UT)")
#         ax.set_ylabel("Average Intensity")
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
#         plt.xticks(rotation=30)
#         fig.tight_layout()
        
#         source_filename = os.path.basename(peak["source_file"])
#         tiff_id = source_filename[3:10]
#         filename = f"F_{tiff_id}_{peak_time.strftime('%H_%M_%S_%f')[:-4]}_{region_idx}.png"
#         save_path = os.path.join(output_folder, filename)
#         fig.savefig(save_path)
#         plt.close(fig)
        
#     print("Plot generation complete.")

def plot_single_pulsation_worker(peak, grid_timeseries, start_time, fps, output_folder):
    """
    Worker function that generates a plot for one individual pulsation event.
    """
    frame_buffer = 50
    region_idx = peak["region"]
    intensity_data = grid_timeseries[region_idx]
    
    start_frame = peak["start_index"]
    end_frame = peak["end_index"]

    plot_start_index = max(0, start_frame - frame_buffer)
    plot_end_index = min(len(intensity_data), end_frame + frame_buffer)
    
    plot_data = intensity_data[plot_start_index:plot_end_index]
    plot_time_objects = [start_time + timedelta(seconds=i/fps) for i in range(plot_start_index, plot_end_index)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_time_objects, plot_data, label=f'Region {region_idx} Intensity', zorder=1.5, linewidth=2)
    
    plot_smoothing_window = 51
    if plot_smoothing_window >= len(plot_data):
        plot_smoothing_window = len(plot_data) - 1
    if plot_smoothing_window % 2 == 0: plot_smoothing_window -= 1
    if plot_smoothing_window > 2:
        smoothed_plot_data = savgol_filter(plot_data, plot_smoothing_window, 3)
        ax.plot(plot_time_objects, smoothed_plot_data, label='Smoothed Data', color='orange', linestyle=':', zorder=2)

    peak_time = start_time + timedelta(seconds=peak["index"] / fps)
    peak_intensity_raw = peak["intensity"] 
    ax.plot(peak_time, peak_intensity_raw, 'x', color='red', markersize=10, mew=2, label=f'Peak', zorder=3)
    
    start_peak_time = start_time + timedelta(seconds=start_frame / fps)
    end_peak_time = start_time + timedelta(seconds=end_frame / fps)
    
    ax.axvline(x=start_peak_time, color='green', linestyle='--', label='Peak Start/End', zorder=1)
    ax.axvline(x=end_peak_time, color='green', linestyle='--', zorder=1)
    
    if peak.get("fwhm_sec", 0) > 0 and peak.get("left_fwhm_idx", -1) != -1:
        peak_idx_local = peak["index"] - plot_start_index
        
        if 'smoothed_plot_data' in locals() and 0 <= peak_idx_local < len(smoothed_plot_data):
            peak_value_smoothed = smoothed_plot_data[peak_idx_local]
            half_height = peak_value_smoothed - (peak["prominence"] / 2)

            left_fwhm_time = start_time + timedelta(seconds=peak["left_fwhm_idx"] / fps)
            right_fwhm_time = start_time + timedelta(seconds=peak["right_fwhm_idx"] / fps)

            ax.hlines(y=half_height, xmin=left_fwhm_time, xmax=right_fwhm_time,
                      color='purple', linestyle='--', label=f'FWHM ({peak["fwhm_sec"]:.2f}s)')
            
            ax.plot([left_fwhm_time, right_fwhm_time], [half_height, half_height], 'o', color='purple')

    ax.set_title(f"Pulsation Detected in Region {region_idx}")
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Average Intensity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    plt.xticks(rotation=30)
    fig.tight_layout()
    
    source_filename = os.path.basename(peak["source_file"])
    tiff_id = source_filename[3:10]
    filename = f"F_{tiff_id}_{peak_time.strftime('%H_%M_%S_%f')[:-4]}_{region_idx}.png"
    save_path = os.path.join(output_folder, filename)
    fig.savefig(save_path)
    plt.close(fig)

def generate_pulsation_plots(final_peaks, grid_timeseries, start_time, fps, output_folder):
    # Goal: Generate a detailed plot for each detected pulsation using parallel processing.
    # Input: final_peaks (list of dict), grid_timeseries (list of np.array), start_time (datetime), fps (float), output_folder (str)
    # Output: None. Saves PNG plot files to the specified output folder.
    if not final_peaks:
        return
        
    print(f"\nGenerating {len(final_peaks)} individual pulsation plots in parallel...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Use partial to pre-fill the arguments that are the same for every plot
    worker_func = partial(plot_single_pulsation_worker, 
                          grid_timeseries=grid_timeseries, 
                          start_time=start_time, 
                          fps=fps, 
                          output_folder=output_folder)
                          
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Map each peak in the final_peaks list to a worker process
        executor.map(worker_func, final_peaks)
        
    print("Individual plot generation complete.")

# --- MAIN PROGRAM ----------------------------------------------------------------------------------------------
def analyze_pulsations_with_grid(folder_path, output_plot_folder, duplicates, find_max_in_bases, generate_plots):
    
    # --- Configurable Parameters ---
    FPS = 56.7
    NORMAL_PROMINENCE = 80
    REDUCED_PROMINENCE = 30
    PIXEL_SIZE = 6
    HIGH_BASE_THRESHOLD = 900
    IMAGE_SIZE = (128, 128)
    GRID_DIMENSIONS = (6, 6)
    SIGMA_CLIP_THRESHOLD = 2 

    # --- Filter Parameters ---
    MAX_PULSATION_SECONDS = 30.0 
    MIN_PROMINENCE_WIDTH_RATIO = 0.15

    # --- Run the full pipeline ---
    start_main_time = time.time()
    
    grid_timeseries_mad, grid_timeseries_raw, frame_to_file_map, num_regions = run_parallel_for_grid(
        folder_path, PIXEL_SIZE, GRID_DIMENSIONS, IMAGE_SIZE, SIGMA_CLIP_THRESHOLD
    )   
    if not grid_timeseries_mad:
        print("Failed to extract time-series data. Exiting.")
        return

    detection_params = (
        NORMAL_PROMINENCE, REDUCED_PROMINENCE, HIGH_BASE_THRESHOLD,
        FPS, find_max_in_bases,
        MAX_PULSATION_SECONDS, MIN_PROMINENCE_WIDTH_RATIO
    )

    all_peaks = detect_peaks_in_grid(grid_timeseries_mad, grid_timeseries_raw, detection_params, frame_to_file_map)
    if not all_peaks:
        print("\nNo initial candidates found to vet. Exiting.")
        return

    vetted_peaks = all_peaks
    
    if duplicates:
        final_peaks = vetted_peaks
        print(f"\nSkipping de-duplication. Keeping all {len(final_peaks)} vetted events.")
    else:
        print(f"\nDe-duplicating the {len(vetted_peaks)} vetted aurora candidates...")
        final_peaks = deduplicate_peaks(vetted_peaks, FPS, GRID_DIMENSIONS)

    log_text = extract_text_from_log(folder_path)
    if log_text:
        start_time = get_start_end_time(log_text)
        
        if final_peaks:
            log_pulsation_data(final_peaks, start_time, FPS, NORMAL_PROMINENCE, PIXEL_SIZE, GRID_DIMENSIONS)
            
            if generate_plots:
                generate_full_timeseries_plots(final_peaks, grid_timeseries_mad, start_time, FPS, output_plot_folder, frame_to_file_map, GRID_DIMENSIONS)
                generate_pulsation_plots(final_peaks, grid_timeseries_mad, start_time, FPS, output_plot_folder)
        else:
            print("\nNo peaks survived the vetting process. No pulsation report will be generated.")
            
    elif not all_peaks:
        print("\nNo peaks met the initial criteria. No report will be generated.")
            
    end_main_time = time.time()
    print(f"\nTotal analysis time: {end_main_time - start_main_time:.2f} seconds.")

if __name__ == '__main__':
    folder = "Big_Data\\Day_1\\ut15"
    output_folder = "pulsation_plots"
    
    duplicates = False
    while True:
        input_str = input("Keep duplicate detections for the same event? (y/n): ").lower()
        if input_str == "y":    
            duplicates = True 
            break
        elif input_str == "n":
            duplicates = False
            break
        else:
            print("Invalid Input. (y/n)")
            
    find_max_in_bases = False
    while True:
        input_str = input("Use absolute max between bases for peak finding? (y/n): ").lower()
        if input_str == "y":    
            find_max_in_bases = True
            print("-> Using absolute max between bases.")
            break
        elif input_str == "n":
            find_max_in_bases = False
            print("-> Using default peak from smoothed data.")
            break
        else:
            print("Invalid Input. (y/n)")

    generate_plots = False
    while True:
        input_str = input("Generate Pulsation Plots? (y/n): ").lower()
        if input_str == "y":    
            generate_plots = True
            break
        elif input_str == "n":
            generate_plots = False
            break
        else:
            print("Invalid Input. (y/n)")

    analyze_pulsations_with_grid(folder, output_folder, duplicates, find_max_in_bases, generate_plots)
