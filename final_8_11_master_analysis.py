import os
import sys
import time
import importlib
from datetime import datetime, timedelta

# =================================================================================================
# ---                              MASTER CONFIGURATION                                         ---
# =================================================================================================
# All user-configurable parameters are located in this section for easy access.

CONFIG = {
    # --- File and Directory Settings ---
    "analysis_script_name": "final_8_11_detection.py",
    "viewer_script_name": "final_8_11_detection_viewer.py",
    "parent_data_directory": "Big_Data",
    "main_output_folder": "analysis_output",

    # --- Batch Analysis Behavior ---
    "keep_duplicates": False,             # If False, combines nearby detections into a single event.
    "generate_individual_plots": True,    # If True, creates a plot for each detected pulsation.
    "generate_full_region_plots": True,   # If True, creates a plot of the full time-series for each region.
    "use_true_peaks": True,               # If True, finds the absolute max intensity in the raw data for the peak.

    # --- Core Detection & Image Parameters ---
    "fps": 56.7,                          # Camera frames per second.
    "image_size": (128, 128),             # Image dimensions (height, width).
    "grid_dimensions": (6, 6),              # Analysis grid size (rows, cols).
    "pixel_size": 6,                      # The side length of the analysis boxes in pixels.
    "sigma_clip_threshold": 2,            # Sigma value for robust outlier rejection during data extraction.

    # --- Peak Finding Parameters ---
    "normal_prominence": 80,              # Prominence required for peaks on a dim background.
    "reduced_prominence": 30,             # Lower prominence threshold for finding all initial candidates.
    "high_base_threshold": 900,           # Intensity level above which a background is considered "bright".
    "max_pulsation_seconds": 30.0,        # Maximum duration in seconds a pulsation can last.
    "min_prominence_width_ratio": 0.15,   # Filters out broad, low-slope events.

    # --- Viewer Parameters ---
    "viewer_buffer_size": 1000,           # Number of frames for the full movie player buffer.
    "viewer_interval_buffer_frames": 100, # Number of frames to show before/after an event in the interval player.
}
# =================================================================================================

# --- DYNAMIC SCRIPT IMPORTS ---
try:
    analysis_module = importlib.import_module(CONFIG["analysis_script_name"].replace('.py', ''))
    viewer_module = importlib.import_module(CONFIG["viewer_script_name"].replace('.py', ''))
except ImportError as e:
    print(f"Error: Could not import from local scripts.")
    print(f"Please ensure '{CONFIG['analysis_script_name']}' and '{CONFIG['viewer_script_name']}' are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)

# --- BATCH ANALYSIS FUNCTIONS ---

def discover_data_directories(parent_dir):
    # Goal: Scan a parent directory to find and organize all day/hour subdirectories.
    # Input: parent_dir (str)
    # Output: A dictionary where keys are day names and values are lists of full paths to hour directories (dict).
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

def custom_log_pulsation_data(output_path, final_peaks, start_time, config):
    # Goal: Write a formatted pulsation report to a file and return the data rows for aggregation.
    # Input: output_path (str), final_peaks (list of dict), start_time (datetime), config (dict)
    # Output: A list of strings, where each string is a data row from the report (list of str).
    output_lines = []
    
    # --- Build Header ---
    output_lines.append(f"Pulsation Detection Report (Method: Savitzky-Golay Filter)")
    output_lines.append(f"Date: {start_time.strftime('%Y-%m-%d')}")
    output_lines.append(f"UT Hour of Start: {start_time.strftime('%H')}")
    output_lines.append("-" * 40)
    output_lines.append(f"Prominence Threshold: {config['normal_prominence']}")
    output_lines.append(f"Pixel Grid Size: {config['pixel_size']}x{config['pixel_size']}")
    
    grid_dims = config['grid_dimensions']
    output_lines.append(f"\nRegion Reference Grid ({grid_dims[0]}x{grid_dims[1]}):")
    region_num = 0
    for r in range(grid_dims[0]):
        line = "  "
        for c in range(grid_dims[1]):
            line += f"{region_num:<3} "
            region_num += 1
        output_lines.append(line)
    output_lines.append("-" * 40)
    
    header = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<15} {:<10} {:<12} {:<14} {:<14} {:<25}".format(
        "Start Time (t1)", "End Time (t2)", "Start Frame", "End Frame", "Peak Frame", 
        "Peak Intensity", "Region", "FWHM (s)", "FWHM Start", "FWHM End", "Source File"
    )
    output_lines.append(header)
    output_lines.append("=" * len(header))
    
    # --- Build Data Rows ---
    data_lines = []
    for peak in final_peaks:
        t1 = start_time + timedelta(seconds=peak["start_index"] / config['fps'])
        t2 = start_time + timedelta(seconds=peak["end_index"] / config['fps'])
        fwhm_start_frame = peak.get("left_fwhm_idx", -1)
        fwhm_end_frame = peak.get("right_fwhm_idx", -1)
        fwhm_start_str = f"{fwhm_start_frame:.0f}" if fwhm_start_frame != -1 else "N/A"
        fwhm_end_str = f"{fwhm_end_frame:.0f}" if fwhm_end_frame != -1 else "N/A"
        line = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<15.2f} {:<10} {:<12.2f} {:<14} {:<14} {:<25}".format(
            t1.strftime('%H:%M:%S.%f')[:-4], t2.strftime('%H:%M:%S.%f')[:-4],
            peak["start_index"], peak["end_index"], peak["index"],
            peak["intensity"], peak["region"], peak["fwhm_sec"],
            fwhm_start_str, fwhm_end_str, os.path.basename(peak["source_file"])
        )
        data_lines.append(line)
    
    output_lines.extend(data_lines)
    
    # --- Write to File ---
    print(f"Writing hourly report to {output_path}...")
    with open(output_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
            
    return data_lines

def run_hourly_analysis(hour_path, hour_output_path, config):
    # Goal: Run the full detection and plotting pipeline for a single hour's data directory.
    # Input: hour_path (str), hour_output_path (str), config (dict)
    # Output: A list of strings containing the data rows for the master report (list of str).
    print("-" * 80)
    print(f"Processing hour: {os.path.basename(hour_path)}")
    start_hour_time = time.time()

    # Stage 1: Data Extraction
    grid_timeseries, frame_to_file_map, _ = analysis_module.run_parallel_for_grid(
        hour_path, config['pixel_size'], config['grid_dimensions'], config['image_size'], config['sigma_clip_threshold']
    )
    if not grid_timeseries:
        print("Failed to extract time-series data. Skipping this hour.")
        return []

    # Stage 2: Peak Detection & Vetting
    detection_params = (
        config['normal_prominence'], config['reduced_prominence'], config['high_base_threshold'],
        config['fps'], config['use_true_peaks'],
        config['max_pulsation_seconds'], config['min_prominence_width_ratio']
    )
    all_peaks = analysis_module.detect_peaks_in_grid(grid_timeseries, detection_params, frame_to_file_map)
    if not all_peaks:
        print("\nNo initial candidates found. Skipping this hour.")
        return []
    
    if config['keep_duplicates']:
        final_peaks = all_peaks
        print(f"\nSkipping de-duplication. Keeping all {len(final_peaks)} events.")
    else:
        print(f"\nDe-duplicating {len(all_peaks)} candidates...")
        final_peaks = analysis_module.deduplicate_peaks(all_peaks, config['fps'], config['grid_dimensions'])

    if not final_peaks:
        print("\nNo peaks survived the vetting process. No report for this hour.")
        return []

    # Stage 3: Reporting & Plotting
    log_text = analysis_module.extract_text_from_log(hour_path)
    if not log_text:
        print("Warning: Could not find .log file in source directory. Cannot generate report or plots.")
        return []
    start_time = analysis_module.get_start_end_time(log_text)
    
    hour_name = os.path.basename(hour_path)
    report_output_path = os.path.join(hour_output_path, f"{hour_name}_report.txt")
    
    report_data_lines = custom_log_pulsation_data(report_output_path, final_peaks, start_time, config)
    
    # Generate plots if requested
    if config['generate_full_region_plots']:
        # **FIX**: Pass the parent directory (`hour_output_path`) directly.
        # The plotting function adds the "full_timeseries_plots" subfolder itself.
        # This prevents creating a nested folder.
        output_folder_for_full_plots = hour_output_path
        analysis_module.generate_full_timeseries_plots(
            final_peaks, grid_timeseries, start_time, config['fps'],
            output_folder_for_full_plots, frame_to_file_map, config['grid_dimensions']
        )
        
    if config['generate_individual_plots']:
        # This function works as expected, so we create the subfolder path here.
        output_folder_for_individual_plots = os.path.join(hour_output_path, "individual_pulsation_plots")
        analysis_module.generate_pulsation_plots(
            final_peaks, grid_timeseries, start_time, config['fps'], output_folder_for_individual_plots
        )

    end_hour_time = time.time()
    print(f"Finished processing {hour_name} in {end_hour_time - start_hour_time:.2f} seconds.")
    return report_data_lines

def run_batch_analysis(config):
    # Goal: Run the analysis pipeline on all discovered data and create a master report.
    # Input: config (dict)
    # Output: None
    print("--- Starting Batch Analysis (Option 1) ---")
    data_structure = discover_data_directories(config['parent_data_directory'])
    if not data_structure:
        print("No data found to analyze. Exiting.")
        return

    os.makedirs(config['main_output_folder'], exist_ok=True)
    master_report_path = os.path.join(config['main_output_folder'], "master_pulsation_report.txt")
    all_pulsations = []

    for day_name, hour_paths in data_structure.items():
        print("="*80)
        print(f"PROCESSING DAY: {day_name}")
        day_output_folder = os.path.join(config['main_output_folder'], f"{day_name}_output")
        os.makedirs(day_output_folder, exist_ok=True)
        for hour_path in hour_paths:
            hour_name = os.path.basename(hour_path)
            hour_output_folder = os.path.join(day_output_folder, f"{hour_name}_output")
            os.makedirs(hour_output_folder, exist_ok=True)
            pulsation_lines = run_hourly_analysis(hour_path, hour_output_folder, config)
            if pulsation_lines:
                all_pulsations.extend(pulsation_lines)
    
    print("="*80)
    print(f"All processing complete. Writing master report to: {master_report_path}")
    header = "{:<20} {:<20} {:<12} {:<12} {:<12} {:<15} {:<10} {:<12} {:<14} {:<14} {:<25}".format(
        "Start Time (t1)", "End Time (t2)", "Start Frame", "End Frame", "Peak Frame", 
        "Peak Intensity", "Region", "FWHM (s)", "FWHM Start", "FWHM End", "Source File"
    )
    with open(master_report_path, 'w') as f:
        f.write("Master Pulsation Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Data: {os.path.abspath(config['parent_data_directory'])}\n")
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write("=" * len(header) + "\n")
        for line in sorted(all_pulsations):
             f.write(line + "\n")
    print("Master report saved successfully.")

# --- INTERACTIVE VIEWER LAUNCHER ---

def launch_viewer(config):
    # Goal: Launch the interactive viewer for a user-specified hour of data.
    # Input: config (dict)
    # Output: None
    print("--- Starting Interactive Viewer (Option 2) ---")
    while True:
        data_folder = input("Please enter the full path to the hour directory you want to view: ").strip()
        if os.path.isdir(data_folder):
            break
        else:
            print(f"Error: Data directory not found at '{data_folder}'. Please try again.")

    try:
        hour_name = os.path.basename(data_folder)
        day_name = os.path.basename(os.path.dirname(data_folder))
        report_path = os.path.join(config['main_output_folder'], f"{day_name}_output", f"{hour_name}_output", f"{hour_name}_report.txt")
        if not os.path.exists(report_path):
            raise FileNotFoundError("Analysis report not found. Please run Batch Analysis (Option 1) first.")

        tiff_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(('.tif', '.tiff'))])
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in '{data_folder}'.")

        frame_map, total_frames = viewer_module.build_frame_map(tiff_files)
        vmin, vmax = viewer_module.calculate_vmin_vmax_by_chunks(tiff_files, chunk_size=2)
        all_intervals, grid_dims = viewer_module.parse_pulsation_report(report_path)
        print(f"Successfully parsed {len(all_intervals)} intervals from '{os.path.basename(report_path)}'.")

        while True:
            mode = input("\nChoose a player mode:\n  1: Full Movie Player\n  2: Individual Pulsation Player\nEnter choice (1 or 2): ")
            if mode in ['1', '2']: break
            print("Invalid choice.")

        if mode == '1':
            player = viewer_module.FullMoviePlayer(
                frame_map, total_frames, vmin, vmax, all_intervals, config['viewer_buffer_size'], 
                grid_dims, config['image_size'], config['pixel_size']
            )
            player.show()
        elif mode == '2':
            if not all_intervals:
                print("No intervals to display.")
                return
            for i, interval in enumerate(all_intervals):
                print(f"Loading interval {i+1}/{len(all_intervals)} (Region {interval['region']})...")
                start = max(0, interval['start_frame'] - config['viewer_interval_buffer_frames'])
                end = min(total_frames, interval['end_frame'] + config['viewer_interval_buffer_frames'])
                frames = viewer_module.load_frames(start, end, frame_map)
                if frames.size > 0:
                    player = viewer_module.IntervalPlayer(
                        frames, vmin, vmax, start, interval, grid_dims, 
                        config['image_size'], config['pixel_size']
                    )
                    player.show()
    except Exception as e:
        print(f"\n--- An Error Occurred in the Viewer ---\n{e}")

# --- MAIN MENU ---

if __name__ == '__main__':
    print("============================================")
    print("=== Pulsation Analysis & Visualization Suite ===")
    print("============================================")
    while True:
        print("\nPlease select an option:")
        print("  1. Run Batch Analysis: Process all data using settings in this script.")
        print("  2. Launch Interactive Viewer: Visually inspect results for a specific hour.")
        print("  Q. Quit")
        choice = input("Enter your choice (1, 2, or Q): ").strip().upper()
        if choice == '1':
            run_batch_analysis(CONFIG)
            break
        elif choice == '2':
            launch_viewer(CONFIG)
            break
        elif choice == 'Q':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or Q.")
