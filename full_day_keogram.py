# import tifffile
# import matplotlib.pyplot as plt
# import time
# import os
# from datetime import datetime
# from concurrent.futures import ProcessPoolExecutor
# import numpy as np
# import matplotlib.dates as mdates
# import psutil
# import matplotlib
# from functools import partial



# matplotlib.use('Agg') 

# def load_folders(folder_path):

#     if not os.path.isdir(folder_path):
#         print(f"Error: Folder not found at '{folder_path}'")
#         exit()

#     file_paths = sorted([
#         os.path.join(folder_path, f)
#         for f in os.listdir(folder_path)
#         if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(folder_path, f))
#     ])

#     if not file_paths:
#         print(f"No TIFF files found in '{folder_path}'")
#         exit() 
#     return file_paths

# def extract_text_from_log(folder_path):

#     if not os.path.isdir(folder_path):
#         return f"Error: The path '{folder_path}' is not a valid directory."
    
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith('.log'):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     print(f"Found log file: {filename}")
#                     return f.read()
#             except Exception as e:
#                 return f"Error reading file '{filename}': {e}"

#     return f"No .log file found in '{folder_path}'."

# def get_start_end_time(log):
#     time_start_index = log.find("TimeStart1")
#     time_end_index = log.find("TimeStop1")
#     date_index = log.find("Date")

#     ut_start_string = log[time_start_index+25:time_start_index+37]
#     ut_end_string = log[time_end_index+25:time_end_index+37]
#     date_str = log[date_index+25:date_index+35]

#     ut_start = datetime.strptime(date_str + " " + ut_start_string, "%Y-%m-%d %H:%M:%S.%f")
#     ut_end = datetime.strptime(date_str + " " + ut_end_string, "%Y-%m-%d %H:%M:%S.%f")

#     # print(ut_end)
#     # print(ut_start)

#     akst_dt_start = ut_start
#     akst_dt_end = ut_end

#     return akst_dt_start, akst_dt_end

# def get_col(file_path, start_col = 61, end_col = 67, increment_by = 1):
#     try:
#         pid = os.getpid()
#         print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
#         image_stack = tifffile.imread(file_path)
#         print(image_stack.shape)
#         subset = image_stack[::increment_by, :, start_col:end_col]
#         return subset
#     except Exception as e:
#         print(f"Error processing {os.path.basename(file_path)}: {e}")
#         return None

# def get_row(file_path, start_row = 61, end_row = 67, increment_by = 1):
#     try:
#         pid = os.getpid()
#         print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
#         image_stack = tifffile.imread(file_path)
#         print(image_stack.shape)
#         subset = image_stack[::increment_by, start_row:end_row, : ]
#         return subset
#     except Exception as e:
#         print(f"Error processing {os.path.basename(file_path)}: {e}")
#         return None

# def run_parallel_folder(file_paths, function, start_index, end_index, increment_by, num_workers): 
#     # The input 'file_paths' is already the complete list of files to process.
#     print(f"\nStarting parallel processing for {len(file_paths)} files...")
#     start_time = time.perf_counter()

#     # REMOVED THIS LINE: file_path = load_folders(file_path)
#     # The list of files is already prepared.
#     worker_func = partial(function, start_index, end_index, increment_by)
#     with ProcessPoolExecutor(max_workers=num_workers) as executor: # Corrected parameter name
#         # Use the 'file_paths' list directly
#         results = list(executor.map(worker_func, file_paths))
    
#     valid_results = [r for r in results if r is not None]
    
#     if not valid_results:
#         print("No files were processed successfully.")
#         return None

#     print("\nCombining results from all processes...")
#     final_array = np.concatenate(valid_results, axis=0)

#     end_time = time.perf_counter()
    
#     print("\n--- Processing Complete ---")
#     print(f"Final array shape: {final_array.shape}")
#     print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
#     return final_array

# def save_keogram(keogram_data, start_time, end_time, day_folder_path, direction):
#     """
#     Generates a keogram plot and saves it directly to a file without displaying it.
#     """
#     print("\nGenerating and saving the final keogram plot...")
#     fig, ax = plt.subplots(figsize=(22, 8)) # Wider figure for a full day

#     # --- Data Plotting (no changes here) ---
#     start_num = mdates.date2num(start_time)
#     end_num = mdates.date2num(end_time)
#     vmin = np.percentile(keogram_data, 5)
#     vmax = np.percentile(keogram_data, 99.8)

#     im = ax.imshow(
#         keogram_data,
#         aspect='auto',
#         cmap='turbo',
#         extent=[start_num, end_num, keogram_data.shape[0], 0],
#         vmin=vmin,
#         vmax=vmax
#     )
    
#     # --- Dynamic Tick Mark Logic ---
#     duration_hours = (end_time - start_time).total_seconds() / 3600
#     if duration_hours > 4:
#         interval = max(1, round(duration_hours / 12))
#         locator = mdates.HourLocator(interval=interval)
#         formatter = mdates.DateFormatter('%H:%M')
#         ax.set_xlabel(f"Time (UT) - Ticks every {interval} hour(s)")
#     else:
#         duration_minutes = duration_hours * 60
#         interval = max(5, round(duration_minutes / 12))
#         locator = mdates.MinuteLocator(interval=interval)
#         formatter = mdates.DateFormatter('%H:%M')
#         ax.set_xlabel(f"Time (UT) - Ticks every {interval} minute(s)")
    
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

#     # --- Labels and Title ---
#     day_folder_name = os.path.basename(day_folder_path)
#     ax.set_ylabel("Spatial Slice Index")
#     ax.set_title(f"Full Day Keogram ({direction}) for {day_folder_name}")
#     fig.colorbar(im, ax=ax, label="Intensity")
#     fig.tight_layout()

#     # --- Filename and Saving ---
#     output_filename = f"{day_folder_name}_full_keogram_{direction}.png"
    
#     print(f"Saving plot to: {output_filename}")
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
#     # IMPORTANT: Close the figure to free up memory
#     plt.close(fig)

# def build_mega_keogram(RAM_FOR_LARGE_TIFF, day_folder_path, direction, start_index, end_index, increment_by):
#     all_items = os.listdir(day_folder_path)
#     hour_directories = []

#     # Get all Hour File Paths
#     try:
#         for item in all_items:
#             if  os.path.isdir(os.path.join(day_folder_path, item)):
#                 hour_directories.append(item)

#     except FileNotFoundError:
#         print("No directories in current file_path")

#     sorted_hour_directories = sorted(hour_directories)

#     # Get all tiff_file_paths
#     individual_tiffs = []
#     for hour in sorted_hour_directories:
#         tiff_hour_paths = load_folders(os.path.join(day_folder_path, hour))
#         individual_tiffs.extend(tiff_hour_paths)

#     # Calculate num_workers for parallel processing
#     available_ram_GB = psutil.virtual_memory().available / (1024**3)
#     print(available_ram_GB)
#     core_count = os.cpu_count()

#     num_parallel_hours = int(available_ram_GB/RAM_FOR_LARGE_TIFF)

#     num_workers = min(core_count , num_parallel_hours)

#     raw_keogram_np = run_parallel_folder(individual_tiffs , get_col, start_index, end_index, increment_by, num_workers=num_workers) if direction == "NS" else run_parallel_folder(individual_tiffs , get_row, num_workers=num_workers)
    
#     transposed_keogram_data = raw_keogram_np.transpose(1, 0, 2) if direction == "NS" else raw_keogram_np.transpose(2, 0, 1)
#     transposed_keogram_data = transposed_keogram_data.reshape(128, -1)

#     log_file_start = extract_text_from_log(os.path.join(day_folder_path,sorted_hour_directories[0]))
#     log_file_end = extract_text_from_log(os.path.join(day_folder_path,sorted_hour_directories[-1]))
#     start_time, x = get_start_end_time(log_file_start)
#     y, end_time = get_start_end_time(log_file_end)

#     save_keogram(transposed_keogram_data, start_time, end_time, day_folder_path, direction)


# def main():

#     RAM_FOR_LARGE_TIFF = 2
#     start_index = 64
#     end_index = 65
#     increment_by = 20
#     direction = "EW"
#     direction = "NS"
#     print(build_mega_keogram(RAM_FOR_LARGE_TIFF, "Big_Data\\Day_1", direction, start_index, end_index, increment_by))


# if __name__ == "__main__":
#     main()

import tifffile
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.dates as mdates
import psutil
import matplotlib

# Use a backend that doesn't require a GUI, essential for saving plots
matplotlib.use('Agg') 

def load_folders(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        exit()
    file_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(folder_path, f))
    ])
    if not file_paths:
        print(f"No TIFF files found in '{folder_path}'")
        exit() 
    return file_paths

def extract_text_from_log(folder_path):
    if not os.path.isdir(folder_path):
        return f"Error: The path '{folder_path}' is not a valid directory."
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.log'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f"Found log file: {filename}")
                    return f.read()
            except Exception as e:
                return f"Error reading file '{filename}': {e}"
    return f"No .log file found in '{folder_path}'."

def get_start_end_time(log):
    time_start_index = log.find("TimeStart1")
    time_end_index = log.find("TimeStop1")
    date_index = log.find("Date")
    ut_start_string = log[time_start_index+25:time_start_index+37]
    ut_end_string = log[time_end_index+25:time_end_index+37]
    date_str = log[date_index+25:date_index+35]
    ut_start = datetime.strptime(date_str + " " + ut_start_string, "%Y-%m-%d %H:%M:%S.%f")
    ut_end = datetime.strptime(date_str + " " + ut_end_string, "%Y-%m-%d %H:%M:%S.%f")
    return ut_start, ut_end

def get_col(file_path, start_col=61, end_col=67, increment_by=1):
    try:
        image_stack = tifffile.imread(file_path)
        # Slice the frames, height, and desired column width
        subset = image_stack[::increment_by, :, start_col:end_col]
        # Average the columns to get a 2D keogram strip (frames, height)
        keogram_strip = np.mean(subset, axis=2)
        return keogram_strip
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def get_row(file_path, start_row=61, end_row=67, increment_by=1):
    try:
        image_stack = tifffile.imread(file_path)
        # Slice the frames, desired row height, and width
        subset = image_stack[::increment_by, start_row:end_row, :]
        # Average the rows to get a 2D keogram strip (frames, width)
        keogram_strip = np.mean(subset, axis=1)
        return keogram_strip
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None
    
def process_single_file_for_keogram(args):
    """
    Top-level helper function for parallel processing to avoid lambda pickling issues.
    It unpacks a tuple of arguments and calls the target worker (get_col or get_row).
    """
    # Unpack all arguments that were passed in the tuple
    target_function, file_path, start_index, end_index, increment_by = args
    
    # Call the actual worker function with the unpacked arguments
    return target_function(file_path, start_index, end_index, increment_by)

def run_parallel_folder(file_paths, function, start_index, end_index, increment_by, num_workers): 
    print(f"\nStarting parallel processing for {len(file_paths)} files...")
    start_time = time.perf_counter()

    # **FIX**: Prepare a list of argument tuples for our new top-level worker function.
    # This avoids the lambda pickling error.
    tasks = [
        (function, path, start_index, end_index, increment_by) 
        for path in file_paths
    ]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the single, top-level helper function over the list of tasks
        results = list(executor.map(process_single_file_for_keogram, tasks))
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No files were processed successfully.")
        return None

    print("\nCombining results from all processes...")
    final_array = np.concatenate(valid_results, axis=0)

    end_time = time.perf_counter()
    print("\n--- Processing Complete ---")
    print(f"Final array shape: {final_array.shape}")
    print(f"Loading Data Time: {end_time - start_time:.2f} seconds")
    return final_array

def save_keogram(keogram_data, start_time, end_time, day_folder_path, direction):
    print("\nGenerating and saving the final keogram plot...")
    start_time_timer = time.perf_counter()

    fig, ax = plt.subplots(figsize=(22, 8))

    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)
    vmin = np.percentile(keogram_data, 5)
    vmax = np.percentile(keogram_data, 99.8)

    im = ax.imshow(
        keogram_data,
        aspect='auto',
        cmap='turbo',
        extent=[start_num, end_num, keogram_data.shape[1], 0], # extent now uses the second dimension
        vmin=vmin,
        vmax=vmax
    )
    
    duration_hours = (end_time - start_time).total_seconds() / 3600
    if duration_hours > 4:
        interval = max(1, round(duration_hours / 12))
        locator = mdates.HourLocator(interval=interval)
        formatter = mdates.DateFormatter('%H:%M')
        ax.set_xlabel(f"Time (UT) - Ticks every {interval} hour(s)")
    else:
        duration_minutes = duration_hours * 60
        interval = max(5, round(duration_minutes / 12))
        locator = mdates.MinuteLocator(interval=interval)
        formatter = mdates.DateFormatter('%H:%M')
        ax.set_xlabel(f"Time (UT) - Ticks every {interval} minute(s)")
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    day_folder_name = os.path.basename(day_folder_path)
    ax.set_ylabel("Spatial Slice Index")
    ax.set_title(f"Full Day Keogram ({direction}) for {day_folder_name}")
    fig.colorbar(im, ax=ax, label="Intensity")
    fig.tight_layout()

    output_filename = f"{day_folder_name}_full_keogram_{direction}.png"
    print(f"Saving plot to: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    end_time_timer = time.perf_counter()
    print(f"Loading Data Time: {end_time_timer - start_time_timer:.2f} seconds")




def build_mega_keogram(RAM_FOR_LARGE_TIFF, day_folder_path, direction, start_index, end_index, increment_by):
    all_items = os.listdir(day_folder_path)
    hour_directories = []
    try:
        for item in all_items:
            if os.path.isdir(os.path.join(day_folder_path, item)):
                hour_directories.append(item)
    except FileNotFoundError:
        print("No directories in current file_path")

    sorted_hour_directories = sorted(hour_directories)

    individual_tiffs = []
    for hour in sorted_hour_directories:
        tiff_hour_paths = load_folders(os.path.join(day_folder_path, hour))
        individual_tiffs.extend(tiff_hour_paths)

    available_ram_GB = psutil.virtual_memory().available / (1024**3)
    core_count = os.cpu_count()
    num_parallel_hours = int(available_ram_GB / RAM_FOR_LARGE_TIFF)
    num_workers = min(core_count, num_parallel_hours)
    print(f"Using {num_workers} parallel workers based on available RAM and CPU cores.")
    
    # **FIX**: Pass all required arguments to both function calls.
    if direction == "NS":
        raw_keogram_np = run_parallel_folder(individual_tiffs, get_col, start_index, end_index, increment_by, num_workers=num_workers)
    else: # "EW"
        raw_keogram_np = run_parallel_folder(individual_tiffs, get_row, start_index, end_index, increment_by, num_workers=num_workers)
    
    if raw_keogram_np is None:
        print("Keogram generation failed because no data was processed.")
        return

    # **FIX**: A keogram is (space, time), so we just need to transpose the final array.
    keogram_data = raw_keogram_np.T

    log_file_start = extract_text_from_log(os.path.join(day_folder_path, sorted_hour_directories[0]))
    log_file_end = extract_text_from_log(os.path.join(day_folder_path, sorted_hour_directories[-1]))
    start_time, _ = get_start_end_time(log_file_start)
    _, end_time = get_start_end_time(log_file_end)

    save_keogram(keogram_data, start_time, end_time, day_folder_path, direction)

def main():
    RAM_FOR_LARGE_TIFF = 2
    start_index = 64
    end_index = 65
    increment_by = 10
    
    # Process one direction at a time
    print("\n--- Building NS Keogram ---")
    build_mega_keogram(RAM_FOR_LARGE_TIFF, "Big_Data/Day_1", "NS", start_index, end_index, increment_by)
    
    print("\n--- Building EW Keogram ---")
    build_mega_keogram(RAM_FOR_LARGE_TIFF, "Big_Data/Day_1", "EW", start_index, end_index, increment_by)

if __name__ == "__main__":
    main()