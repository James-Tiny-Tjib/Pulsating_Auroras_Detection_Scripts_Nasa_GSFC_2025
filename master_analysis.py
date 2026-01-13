"""
Title: Pulsating Aurora Auto-Detection Module
Author: James Hui
Affiliation: River Hill High School, UMD
Email: james.y.hui@gmail.com
GitHub: https://github.com/James-Tiny-Tjib
Date Created: 2025-01-06
Last Updated: 2025-01-08
"""

import os
import sys
import shutil
import time
from datetime import datetime, timedelta


# --- Import functions from your existing scripts ---
try:
    from detection import (
        run_parallel_for_grid, detect_peaks_in_grid, deduplicate_peaks,
        extract_text_from_log, get_start_end_time, generate_pulsation_plots,
        generate_full_timeseries_plots
    )
    from detection_viewer import (
        build_frame_map as viewer_build_frame_map,
        calculate_vmin_vmax as calculate_vmin_vmax, 
        parse_pulsation_report,
        FullMoviePlayer,
        IntervalPlayer,
        load_frames,
        save_pulsation_video
    )
except ImportError as e:
    print(f"Error: Could not import from local scripts.") 
    print(f"Please ensure 'final_8_1_detection.py' and 'final_8_1_detection_viewer.py' are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)

# =================================================================================================
# --- GLOBAL CONFIGURATION ---
# =================================================================================================
PARENT_DATA_DIRECTORY = "Big_Data"  # <-- SET YOUR MAIN DATA FOLDER PATH HERE
MAIN_OUTPUT_FOLDER = "analysis_output"

CONFIG = {
    "keep_duplicates": False,
    "generate_individual_plots": True,
    "generate_full_region_plots": True,
    "use_true_peaks": True
}
# =================================================================================================

def discover_data_directories(parent_dir):
    """Scans the parent directory to find and organize day and hour subdirectories."""
    if not os.path.isdir(parent_dir):
        print(f"Error: The specified parent data directory does not exist: {parent_dir}")
        return {}

    data_structure = {}
    print(f"Scanning for data in: {parent_dir}")
    day_dirs = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])

    for day in day_dirs:
        day_path = os.path.join(parent_dir, day)
        hour_dirs = sorted([h for h in os.listdir(day_path) if os.path.isdir(os.path.join(day_path, h))])
        
        if hour_dirs:
            data_structure[day] = [os.path.join(day_path, hour) for hour in hour_dirs]

    print(f"Found {len(data_structure)} day(s) with hourly data.")
    return data_structure

def custom_log_pulsation_data(output_path, final_peaks, start_time, fps, prominence, pixel_size, grid_dims):
    """Writes the pulsation report to a specified file path and returns the content."""
    output_lines = []
    
    # --- Build Header ---
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
    
    # --- Updated Header ---
    header = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<12} {:<12} {:<10} {:<12} {:<14} {:<14} {:<25}".format(
        "Start Time (t1)", "End Time (t2)", "Start Frame", "End Frame", "Peak Frame", 
        "Int (MAD)", "Int (Raw)", "Region", "FWHM (s)", "FWHM Start", "FWHM End", "Source File"
    )
    output_lines.append(header)
    output_lines.append("=" * len(header))
    
    # --- Build Data Rows ---
    data_lines = []
    for peak in final_peaks:
        t1 = start_time + timedelta(seconds=peak["start_index"] / fps)
        t2 = start_time + timedelta(seconds=peak["end_index"] / fps)
        
        fwhm_start_frame = peak.get("left_fwhm_idx", -1)
        fwhm_end_frame = peak.get("right_fwhm_idx", -1)
        fwhm_start_str = f"{fwhm_start_frame:.0f}" if fwhm_start_frame != -1 else "N/A"
        fwhm_end_str = f"{fwhm_end_frame:.0f}" if fwhm_end_frame != -1 else "N/A"

        line = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<12.2f} {:<12.2f} {:<10} {:<12.2f} {:<14} {:<14} {:<25}".format(
            t1.strftime('%H:%M:%S.%f')[:-4], t2.strftime('%H:%M:%S.%f')[:-4],
            peak["start_index"], peak["end_index"], peak["index"],
            peak.get("intensity_mad", peak.get("intensity", 0)), 
            peak.get("intensity_raw", 0), 
            peak["region"], 
            peak["fwhm_sec"],
            fwhm_start_str, fwhm_end_str,
            os.path.basename(peak["source_file"])
        )
        data_lines.append(line)
    
    output_lines.extend(data_lines)
    
    # --- Write to File ---
    print(f"Writing hourly report to {output_path}...")
    with open(output_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
            
    return data_lines

def run_hourly_analysis(hour_path, day_output_path, config):
    """
    Runs the full detection and analysis pipeline for a single hour.
    Outputs are saved directly into the Day folder (day_output_path).
    """
    print("-" * 80)
    print(f"Processing hour: {os.path.basename(hour_path)}")
    start_hour_time = time.time()

    # --- Configurable Parameters ---
    FPS = 56.7
    NORMAL_PROMINENCE = 80
    REDUCED_PROMINENCE = 30
    PIXEL_SIZE = 6
    HIGH_BASE_THRESHOLD = 900
    IMAGE_SIZE = (128, 128)
    GRID_DIMENSIONS = (6, 6)
    SIGMA_CLIP_THRESHOLD = 2
    MAX_PULSATION_SECONDS = 90.0
    MIN_PROMINENCE_WIDTH_RATIO = 0.15

    # STAGE 1: DATA EXTRACTION
    grid_timeseries_mad, grid_timeseries_raw, frame_to_file_map, num_regions = run_parallel_for_grid(
        hour_path, PIXEL_SIZE, GRID_DIMENSIONS, IMAGE_SIZE, SIGMA_CLIP_THRESHOLD
    )
    if not grid_timeseries_mad:
        print("Failed to extract time-series data. Skipping this hour.")
        return []

    # STAGE 2: PEAK DETECTION & VETTING
    detection_params = (
        NORMAL_PROMINENCE, REDUCED_PROMINENCE, HIGH_BASE_THRESHOLD,
        FPS, config['use_true_peaks'],
        MAX_PULSATION_SECONDS, MIN_PROMINENCE_WIDTH_RATIO
    )
    
    all_peaks = detect_peaks_in_grid(grid_timeseries_mad, grid_timeseries_raw, detection_params, frame_to_file_map)
    
    if not all_peaks:
        print("\nNo initial candidates found to vet. Skipping this hour.")
        return []
    
    if config['keep_duplicates']:
        final_peaks = all_peaks
        print(f"\nSkipping de-duplication. Keeping all {len(final_peaks)} vetted events.")
    else:
        print(f"\nDe-duplicating the {len(all_peaks)} vetted aurora candidates...")
        final_peaks = deduplicate_peaks(all_peaks, FPS, GRID_DIMENSIONS)

    if not final_peaks:
        print("\nNo peaks survived the vetting process. No report for this hour.")
        return []

    # STAGE 3: REPORTING & PLOTTING
    log_text = extract_text_from_log(hour_path)
    if not log_text:
        print("Warning: Could not find .log file. Cannot generate report or plots.")
        return []

    start_time = get_start_end_time(log_text)
    
    # --- SAVE TO DAY FOLDER DIRECTLY ---
    hour_name = os.path.basename(hour_path)
    report_output_path = os.path.join(day_output_path, f"{hour_name}_report.txt")
    
    report_data_lines = custom_log_pulsation_data(
        report_output_path, final_peaks, start_time, FPS, NORMAL_PROMINENCE, PIXEL_SIZE, GRID_DIMENSIONS
    )
    
    # Plots go into shared subfolders in the day directory
    if config['generate_full_region_plots']:
        output_folder = os.path.join(day_output_path, "full_region_plots")
        os.makedirs(output_folder, exist_ok=True)
        generate_full_timeseries_plots(final_peaks, grid_timeseries_mad, start_time, FPS, output_folder, frame_to_file_map, GRID_DIMENSIONS)
        
    if config['generate_individual_plots']:
        output_folder = os.path.join(day_output_path, "individual_pulsation_plots")
        os.makedirs(output_folder, exist_ok=True)
        generate_pulsation_plots(final_peaks, grid_timeseries_mad, start_time, FPS, output_folder)

    end_hour_time = time.time()
    print(f"Finished processing {hour_name} in {end_hour_time - start_hour_time:.2f} seconds.")
    return report_data_lines

def run_batch_analysis(parent_dir, main_output_dir, config):
    """Main function for Option 1."""
    print("--- Starting Batch Analysis (Option 1) ---")
    data_structure = discover_data_directories(parent_dir)
    if not data_structure:
        print("No data found to analyze. Exiting.")
        return

    os.makedirs(main_output_dir, exist_ok=True)
    master_report_path = os.path.join(main_output_dir, "master_pulsation_report.txt")
    all_pulsations = []

    for day_name, hour_paths in data_structure.items():
        print("="*80)
        print(f"PROCESSING DAY: {day_name}")
        day_output_folder = os.path.join(main_output_dir, f"{day_name}_output")
        os.makedirs(day_output_folder, exist_ok=True)

        for hour_path in hour_paths:
            # Pass the Day Output Folder directly. No sub-folder creation here.
            pulsation_lines = run_hourly_analysis(hour_path, day_output_folder, config)
            if pulsation_lines:
                all_pulsations.extend(pulsation_lines)
    
    print("="*80)
    print(f"All processing complete. Writing master report to: {master_report_path}")
    header = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<12} {:<12} {:<10} {:<12} {:<14} {:<14} {:<25}".format(
        "Start Time (t1)", "End Time (t2)", "Start Frame", "End Frame", "Peak Frame", 
        "Int (MAD)", "Int (Raw)", "Region", "FWHM (s)", "FWHM Start", "FWHM End", "Source File"
    )
    with open(master_report_path, 'w') as f:
        f.write("Master Pulsation Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Data: {os.path.abspath(parent_dir)}\n")
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write("=" * len(header) + "\n")
        for line in sorted(all_pulsations):
             f.write(line + "\n")
    
    print("Master report saved successfully.")

def launch_viewer():
    """Main function for Option 2."""
    print("--- Starting Interactive Viewer (Option 2) ---")
    
    while True:
        data_folder = input("Please enter the full path to the hour directory you want to view (e.g., Big_Data/Day_1/ut01): ").strip()
        if os.path.isdir(data_folder):
            break
        else:
            print(f"Error: Data directory not found at '{data_folder}'. Please try again.")

    try:
        hour_name = os.path.basename(data_folder)
        day_name = os.path.basename(os.path.dirname(data_folder))
        
        # Look directly in the Day folder for the report
        report_path = os.path.join(MAIN_OUTPUT_FOLDER, f"{day_name}_output", f"{hour_name}_report.txt")
        print(f"Looking for report file at: {report_path}")

        if not os.path.exists(report_path):
            print("\n--- WARNING ---")
            print("The analysis report for this hour was not found in the output folder.")
            print("Please ensure you have run the Batch Analysis (Option 1) for this dataset first.")
            return

        CONFIRMED_REPORT_FILE = report_path
    except Exception as e:
        print(f"Error: Could not determine the path to the report file. Details: {e}")
        return

    # Constants
    BUFFER_SIZE = 1000
    IMAGE_SIZE = (128, 128)
    PIXEL_SIZE = 6
    FPS = 56.7

    try:
        # Load Data
        tiff_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(('.tif', '.tiff'))])
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in '{data_folder}'.")

        log_text = extract_text_from_log(data_folder)
        session_start_time = get_start_end_time(log_text)

        frame_map, total_frames = viewer_build_frame_map(tiff_files)
        vmin, vmax = calculate_vmin_vmax(tiff_files)
        
        all_intervals, grid_dims, PIXEL_SIZE = parse_pulsation_report(CONFIRMED_REPORT_FILE)
        print(f"Successfully parsed {len(all_intervals)} confirmed intervals from '{os.path.basename(CONFIRMED_REPORT_FILE)}'.")
        print(f"Detected Pixel Size: {PIXEL_SIZE}x{PIXEL_SIZE}")

        while True:
            # --- UPDATED MENU ---
            print("\nChoose a player mode:")
            print("  1: Full Movie Player")
            print("  2: Individual Pulsation Player")
            print("  3: Export Pulsation to Video (.mp4)") # New Option
            print("  Q: Quit Viewer")
            
            mode = input("Enter choice: ").strip().upper()
            
            if mode == 'Q':
                break

            if mode == '1':
                player = FullMoviePlayer(
                    frame_map=frame_map, total_frames=total_frames, vmin=vmin, vmax=vmax,
                    all_intervals=all_intervals, 
                    buffer_size=BUFFER_SIZE, grid_dims=grid_dims, image_size=IMAGE_SIZE, pixel_size=PIXEL_SIZE,
                    session_start_time=session_start_time, fps=FPS
                )
                player.show()

            elif mode == '2':
                if not all_intervals:
                    print(f"No confirmed intervals to display.")
                    continue

                BUFFER_FRAMES = 100
                for i, interval in enumerate(all_intervals):
                    print(f"Loading interval {i+1}/{len(all_intervals)} (Region {interval['region']})...")
                    display_start = max(0, interval['start_frame'] - BUFFER_FRAMES)
                    display_end = min(total_frames, interval['end_frame'] + BUFFER_FRAMES)
                    interval_frames = load_frames(display_start, display_end, frame_map)
                    
                    if interval_frames.size > 0:
                        player = IntervalPlayer(
                            frames=interval_frames, vmin=vmin, vmax=vmax, global_start_frame=display_start,
                            interval_data=interval, grid_dims=grid_dims, image_size=IMAGE_SIZE,
                            pixel_size=PIXEL_SIZE, fps=FPS
                        )
                        player.show()
            
            elif mode == '3':
                start_frame_input = input("Enter the Start Frame of the pulsation (from report): ").strip()
                if not start_frame_input.isdigit():
                    print("Invalid input. Please enter a number.")
                    continue
                start_frame_req = int(start_frame_input)
                
                # Find the specific interval
                target_interval = next((p for p in all_intervals if p['start_frame'] == start_frame_req), None)
                
                if not target_interval:
                    print(f"No pulsation found starting at frame {start_frame_req}.")
                    continue

                try:
                    pre_roll = int(input("How many frames BEFORE the start frame? "))
                    post_roll = int(input("How many frames AFTER the end frame? "))
                    show_boxes = input("Show boxes in video? (y/n): ").lower() == 'y'
                    
                    # --- NEW: ASK FOR SPEED ---
                    speed_input = input("Enter playback speed (e.g., 1, 2, 4, 10): ").strip()
                    playback_speed = int(speed_input) if speed_input.isdigit() and int(speed_input) > 0 else 1
                    
                    # Create Output Folder
                    video_dir = os.path.join(MAIN_OUTPUT_FOLDER, f"{day_name}_output", "Saved_videos")
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # Call function with new 'playback_speed' argument
                    save_pulsation_video(
                        video_dir, target_interval, frame_map, vmin, vmax, grid_dims, 
                        IMAGE_SIZE, PIXEL_SIZE, session_start_time, FPS,
                        pre_roll, post_roll, show_boxes, playback_speed
                    )
                except ValueError:
                    print("Invalid number entered. Export cancelled.")

    except Exception as e:
        print(f"\n--- An Error Occurred in the Viewer ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=====================================================")
    print("=== Pulsation Analysis & Visualization Suite      ===")
    print("=====================================================")

    while True:
        print("\nPlease select an option:")
        print("  1. Run Batch Analysis: Find pulsations and generate plots for all data.")
        print("  2. Launch Interactive Viewer: Visually inspect results for a specific hour.")
        print("  Q. Quit")
        
        choice = input("Enter your choice (1, 2, or Q): ").strip().upper()

        if choice == '1':
            run_batch_analysis(PARENT_DATA_DIRECTORY, MAIN_OUTPUT_FOLDER, CONFIG)
            break
        elif choice == '2':
            launch_viewer()
            break
        elif choice == 'Q':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or Q.")

