import tifffile
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.dates as mdates
from pynput import keyboard
import threading





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

    # print(ut_end)
    # print(ut_start)

    akst_dt_start = ut_start # - timedelta(hours=9)
    akst_dt_end = ut_end # - timedelta(hours=9)

    return akst_dt_start, akst_dt_end

def get_col_6166(file_path):
    try:
        pid = os.getpid()
        print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
        image_stack = tifffile.imread(file_path)
        print(image_stack.shape)
        subset = image_stack[:, :, 61:67]
        return subset
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def get_row_6166(file_path):
    try:
        pid = os.getpid()
        print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
        image_stack = tifffile.imread(file_path)
        print(image_stack.shape)
        subset = image_stack[:, 61:67, : ]
        return subset
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def get_all_images(file_path):
    try:
        pid = os.getpid()
        print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
        image_stack = tifffile.imread(file_path)
        return image_stack
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def run_parallel_folder(file_path, function, npArr = None):
    print(f"\nStarting parallel processing for {len(file_path)} files...")
    start_time = time.perf_counter()

    file_path = load_folders(file_path)
    print(file_path)

    if (npArr is not None):
        return get_row_col_6166_6166(file_path[0], npArr)
        
    else:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(function, file_path))
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("No files were processed successfully.")
            return None

        print("\nCombining results from all processes...")
        final_array = np.concatenate(valid_results, axis=0)

        end_time = time.perf_counter()
        
        print("\n--- Processing Complete ---")
        print(f"Final array shape: {final_array.shape}")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
        return final_array

def plot_keogram(keogram_data, start_time, end_time):
    """
    Plots a keogram with a properly formatted time axis.

    Args:
        keogram_data (np.ndarray): The 2D keogram array (e.g., shape (128, N)).
        start_time (datetime): The start time of the observation.
        end_time (datetime): The end time of the observation.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Convert datetime objects to Matplotlib's internal float format
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)

    # Display the keogram data
    # extent defines the coordinates of the image: [left, right, bottom, top]
    # The y-extent is set to (height, 0) to place the 0th row at the top.
    im = ax.imshow(
        keogram_data,
        aspect='auto',
        cmap='viridis',
        extent=[start_num, end_num, keogram_data.shape[0], 0],
        vmin=100,
        vmax=1100
        )

    # --- Configure the X-axis (Time) ---
    # Set the locator to place a tick every 30 minutes
    locator = mdates.MinuteLocator(interval=5)
    ax.xaxis.set_major_locator(locator)

    # Set the formatter to display time as HH:MM
    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)

    # Set labels and title
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Spatial Slice Index")
    ax.set_title("Keogram")
    fig.colorbar(im, ax=ax, label="Intensity")
    
    # Ensure the plot layout is tight
    fig.tight_layout()
    plt.show()

# def plot_both_keograms(ns_keogram_data, ew_keogram_data, start_time, end_time, return_min_max = False):
#     """
#     Plots the North-South and East-West keograms in a single window
#     with a shared, dynamic time axis.
#     """
#     if return_min_max:
#         vmin_ns = np.percentile(ns_keogram_data,5)
#         vmax_ns = np.percentile(ns_keogram_data, 95)
#         return vmin_ns, vmax_ns
#     # 1. Create a figure with 2 vertically stacked subplots that share an X-axis
#     fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

#     # Convert datetime objects to Matplotlib's numerical format
#     start_num = mdates.date2num(start_time)
#     end_num = mdates.date2num(end_time)

#     vmin_ns = np.percentile(ns_keogram_data,5)
#     vmax_ns = np.percentile(ns_keogram_data, 95)

#     vmin_ew = np.percentile(ew_keogram_data, 5)
#     vmax_ew = np.percentile(ew_keogram_data, 95)

#     # --- Plot on the TOP subplot (North-South) ---
#     im_ns = axes[0].imshow(
#         ns_keogram_data,
#         aspect='auto',
#         cmap='turbo',
#         extent=[start_num, end_num, ns_keogram_data.shape[0], 0],
#         vmin=vmin_ns, vmax=vmax_ns
#     )
#     axes[0].set_title("North-South Keogram")
#     axes[0].set_ylabel("Spatial Slice Index")
#     fig.colorbar(im_ns, ax=axes[0], label="Intensity")

#     # --- Plot on the BOTTOM subplot (East-West) ---
#     im_ew = axes[1].imshow(
#         ew_keogram_data,
#         aspect='auto',
#         cmap='turbo',
#         extent=[start_num, end_num, ew_keogram_data.shape[0], 0],
#         vmin=vmin_ew, vmax=vmax_ew
#     )
#     axes[1].set_title("East-West Keogram")
#     axes[1].set_ylabel("Spatial Slice Index")
#     fig.colorbar(im_ew, ax=axes[1], label="Intensity")

#     # --- Configure the SHARED X-axis (Time) ---
#     duration_seconds = (end_time - start_time).total_seconds()
#     if duration_seconds <= 600:
#         interval_sec = max(1, round(duration_seconds / 10))
#         locator = mdates.SecondLocator(interval=int(interval_sec))
#         formatter = mdates.DateFormatter('%H:%M:%S')
#     else:
#         interval_min = max(1, round((duration_seconds / 60) / 12))
#         locator = mdates.MinuteLocator(interval=int(interval_min))
#         formatter = mdates.DateFormatter('%H:%M')
    
#     # Apply locator and formatter to the shared axis
#     axes[1].xaxis.set_major_locator(locator)
#     axes[1].xaxis.set_major_formatter(formatter)
#     axes[1].set_xlabel("Time (UT)")

#     # Add a main title for the entire figure
#     fig.suptitle("Keogram Analysis", fontsize=16)
    
#     plt.tight_layout()
#     plt.show()
def plot_both_keograms(ns_keogram_data, ew_keogram_data, start_time, end_time, return_min_max = False):
    """
    Plots the North-South and East-West keograms in a single window
    with formatting similar to the example provided.
    """
    if return_min_max:
        vmin_ns = np.percentile(ns_keogram_data, 5)
        vmax_ns = np.percentile(ns_keogram_data, 95)
        return vmin_ns, vmax_ns

    # 1. Create a figure with 2 vertically stacked subplots that share an X-axis
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # Convert datetime objects to Matplotlib's numerical format for the extent
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)

    # --- Setup plot parameters ---
    vmin_ns = np.percentile(ns_keogram_data, 5)
    vmax_ns = np.percentile(ns_keogram_data, 95)
    vmin_ew = np.percentile(ew_keogram_data, 5)
    vmax_ew = np.percentile(ew_keogram_data, 95)

    # Define the y-axis range in kilometers
    y_extent = [-3.5, 3.5]

    # --- Plot on the TOP subplot (North-South) ---
    im_ns = axes[0].imshow(
        ns_keogram_data,
        aspect='auto',
        cmap='turbo',
        extent=[start_num, end_num, y_extent[0], y_extent[1]],
        vmin=vmin_ns, vmax=vmax_ns
    )
    axes[0].set_ylabel("Angular Distance (km)")
    fig.colorbar(im_ns, ax=axes[0], label="Counts")
    
    secax_ns = axes[0].secondary_yaxis('right')
    secax_ns.set_yticks([y_extent[1], y_extent[0]])
    secax_ns.set_yticklabels(['North', 'South'], fontsize=10)
    secax_ns.tick_params(axis='y', length=0)

    # --- Plot on the BOTTOM subplot (East-West) ---
    im_ew = axes[1].imshow(
        ew_keogram_data,
        aspect='auto',
        cmap='turbo',
        extent=[start_num, end_num, y_extent[0], y_extent[1]],
        vmin=vmin_ew, vmax=vmax_ew
    )
    axes[1].set_ylabel("Angular Distance (km)")
    fig.colorbar(im_ew, ax=axes[1], label="Counts")

    secax_ew = axes[1].secondary_yaxis('right')
    secax_ew.set_yticks([y_extent[1], y_extent[0]])
    secax_ew.set_yticklabels(['East', 'West'], fontsize=10)
    secax_ew.tick_params(axis='y', length=0)

    # --- Configure the SHARED X-axis (Time) ---
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds <= 600:
        interval_sec = max(1, round(duration_seconds / 10))
        locator = mdates.SecondLocator(interval=int(interval_sec))
        formatter = mdates.DateFormatter('%H:%M:%S')
    else:
        interval_min = max(1, round((duration_seconds / 60) / 12))
        locator = mdates.MinuteLocator(interval=int(interval_min))
        formatter = mdates.DateFormatter('%H:%M')
    
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].set_xlabel("Time (UT)")

    # --- Set the new main title ---
    title_str = f"Keogram Data {start_time.strftime('%Y-%m-%d; %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
    fig.suptitle(title_str, fontsize=16)
    
    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout()
    # Adjust top for suptitle and left/bottom for label padding
    fig.subplots_adjust(left=0.06, bottom=0.1, top=0.94) # <-- MODIFIED LINE

    plt.show()

def graph_images_from_frames(file_paths, start_frame, end_frame, start_time, end_time, vmin, vmax):
    # --- State Variables ---
    state = {
        'is_paused': False,
        'frame_adjustment': 0,
        'should_exit': False
    }
    lock = threading.Lock()

    # --- Nested Listener Function ---
    def on_press(key):
        with lock:
            if key == keyboard.Key.esc:
                state['should_exit'] = True
            elif key == keyboard.Key.space:
                state['is_paused'] = not state['is_paused']
                print("Paused" if state['is_paused'] else "Resumed")
            elif key == keyboard.Key.right:
                state['frame_adjustment'] = 250
            elif key == keyboard.Key.left:
                state['frame_adjustment'] = -250
            elif state['is_paused']:
                if hasattr(key, 'char') and key.char == '.':
                    state['frame_adjustment'] = 1
                elif hasattr(key, 'char') and key.char == ',':
                    state['frame_adjustment'] = -1
    
    # --- Start the Listener ---
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    print("\n--- Video Player Controls ---")
    print("Esc: Close Player | Space: Pause/Resume | Left/Right Arrow: Skip +/- 250 frames")
    print("Period/Comma: Step 1 frame forward/backward (when paused)")

    # --- Pre-scan and setup ---
    frame_counts = [len(tifffile.TiffFile(fp).pages) for fp in file_paths]
    cumulative_counts = np.cumsum(frame_counts)
    total_frames_to_play = end_frame - start_frame
    time_per_frame = (end_time - start_time) / total_frames_to_play if total_frames_to_play > 0 else timedelta()

    plt.ion()
    fig, ax = plt.subplots(); ax.axis('off')
    image_artist = ax.imshow([[0]], cmap='turbo', vmin=vmin, vmax=vmax)
    title_obj = ax.set_title("Initializing..."); plt.show(block=False)

    # --- Main Display Loop ---
    current_tif_file, current_file_index, frame_offset = None, -1, 0
    
    try:
        while not state['should_exit']:
            # Get and apply any requested frame adjustment
            with lock:
                adjustment = state['frame_adjustment']
                state['frame_adjustment'] = 0
            
            if adjustment != 0:
                frame_offset += adjustment
            # Automatically advance frame if not paused and not at the end
            elif not state['is_paused'] and frame_offset < total_frames_to_play - 1:
                frame_offset += 1
            
            # Clamp the frame offset to be within valid bounds [0, last_frame]
            frame_offset = max(0, min(total_frames_to_play - 1, frame_offset))
            
            # Automatically pause if we hit the last frame
            if frame_offset == total_frames_to_play - 1 and not state['is_paused']:
                print("\nVideo finished. Paused at the last frame.")
                state['is_paused'] = True

            # --- Redraw the screen with the current frame ---
            global_frame_idx = start_frame + frame_offset
            target_file_index = np.searchsorted(cumulative_counts, global_frame_idx, side='right')
            local_frame_index = global_frame_idx - (cumulative_counts[target_file_index - 1] if target_file_index > 0 else 0)
            
            if target_file_index != current_file_index:
                if current_tif_file: current_tif_file.close()
                current_tif_file = tifffile.TiffFile(file_paths[target_file_index])
                current_file_index = target_file_index

            frame_data = current_tif_file.pages[local_frame_index].asarray()
            image_artist.set_data(frame_data)
            current_time = start_time + (frame_offset * time_per_frame)
            title_obj.set_text(f"File: {os.path.basename(file_paths[current_file_index])}\n"
                               f"Frame: {global_frame_idx} (Playback Pos: {frame_offset})\n"
                               f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            fig.canvas.draw(); fig.canvas.flush_events()
            
            # A small sleep to prevent the loop from using 100% CPU
            time.sleep(0.5/56.7)
    finally:
        # --- Cleanup ---
        print("Stopping video player and closing window.")
        if current_tif_file: current_tif_file.close()
        listener.stop()
        plt.close(fig)
        plt.ioff()

def get_row_col_6166_6166(file_path, npArr = None):
    if npArr is not None:
        if npArr.shape[1] == 128 and npArr.shape[2] == 6:
            subset = npArr[:, 61:67, :]
            averaged_values = np.mean(subset, axis=(1, 2))
        elif npArr.shape[1] == 6 and npArr.shape[2] == 128:
            subset = npArr[:, :, 61:67]
            averaged_values = np.mean(subset, axis=(1, 2))
        return averaged_values
    
    else:    
    
        try:
            pid = os.getpid()
            print(f"Process ID: {pid} is processing file: {os.path.basename(file_path)}")
            image_stack = tifffile.imread(file_path)
            print(image_stack.shape)
            subset = image_stack[:, 61:67, 61:67]
            print("Subset image_stack:", subset.shape)
            averaged_values = np.mean(subset, axis=(1, 2))
            return averaged_values
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            return None
             
def intensity_time_plot(intensity_time_data, start_time, end_time):
    # 1. Create the figure and axes
    fig, ax = plt.subplots(figsize=(15, 5))

    # 2. Generate the time axis
    # Convert datetime objects to Matplotlib's numerical format
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)
    
    # Create an array of time points matching the length of the data
    time_axis = np.linspace(start_num, end_num, len(intensity_time_data))

    # 3. Plot the data
    ax.plot(time_axis, intensity_time_data, color='#ff00ff', linewidth=1.5)

    # 4. Set Y-axis limits
    if np.max(intensity_time_data)<1000:
        ax.set_ylim(0, 1000)
    else:
        ax.set_ylim(0, np.max(intensity_time_data)+100)

    # 5. Configure the X-axis for time
    # Set the locator to place a tick every 5 minutes
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds <= 600:
        interval_sec = max(1, round(duration_seconds / 10)) # Aim for ~8 ticks
        locator = mdates.SecondLocator(interval=int(interval_sec))
        formatter = mdates.DateFormatter('%H:%M:%S')
    # If duration is over 10 minutes, use Minute ticks
    else:
        interval_min = max(1, round((duration_seconds / 60) / 12)) # Aim for ~10 ticks
        locator = mdates.MinuteLocator(interval=int(interval_min))
        formatter = mdates.DateFormatter('%H:%M')
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # 6. Set labels and title
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Average Intensity")
    ax.set_title("Intensity Over Time")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure the plot layout is tight and display it
    fig.tight_layout()
    plt.ion()
    plt.show(block = False)
    fig.subplots_adjust(left=0.06, bottom=0.1, top=0.94)

def index_from_time(og_start_time, og_end_time, current_time, end_frame_index):
    # duration_seconds = (current_time - og_start_time).total_seconds()
    # total_seconds = (og_end_time-og_start_time).total_seconds()
    # percentage = duration_seconds/total_seconds
    # return int(end_frame_index*percentage)
    duration_seconds = (current_time - og_start_time).total_seconds()
    return int(duration_seconds*56.7)

def data_analyzer(folder_path, skip_keogram = False):

    if skip_keogram:
        og_intensity_time_data = run_parallel_folder(folder_path, get_row_col_6166_6166)
    else:
        og_ns_keo_data = run_parallel_folder(folder_path, get_col_6166)
        og_ew_keo_data = run_parallel_folder(folder_path, get_row_6166)
        og_intensity_time_data = run_parallel_folder(folder_path, get_row_col_6166_6166, og_ns_keo_data)

    og_ns_keo_data = og_ns_keo_data.transpose(1, 0, 2)
    og_ns_keo_data = og_ns_keo_data.reshape(128, -1)

    og_ew_keo_data = og_ew_keo_data.transpose(2, 0, 1)
    og_ew_keo_data = og_ew_keo_data.reshape(128, -1)


    log_file = extract_text_from_log(folder_path)
    og_start_time, og_end_time = get_start_end_time(log_file)
    start_time, end_time = get_start_end_time(log_file)

    start_frame_index = 0
    end_frame_index = len(og_intensity_time_data)

    intensity_time_plot(og_intensity_time_data, og_start_time, og_end_time)
    
    while (True):
        input_str = input("1. Edit Start Time\n2. Edit End Time\n3. View Video \n4. View Keogram\n5. View Timeplot\n6. Reset to Default\n7. Quit\nEnter in your option (1-7): ")
        
        if(input_str == "1"):
            while True:
                time_str = input(f"The Original Start time is {og_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\nEnter a time (HH:MM:SS.ms): ")
                try:
                    time_format = "%H:%M:%S.%f"
                    parsed_time_only = datetime.strptime(time_str, time_format).time()
                    new_parsed_time = datetime.combine(og_start_time.date(),parsed_time_only)
                    if (new_parsed_time >= og_start_time) and (new_parsed_time <= og_end_time) and (new_parsed_time < end_time):
                        print(f"\nSuccess! Created datetime object: {new_parsed_time}")
                        start_time = new_parsed_time
                        start_frame_index = index_from_time(og_start_time, og_end_time, start_time, len(og_intensity_time_data))
                        print(start_frame_index)
                        break
                    else:
                        print(f"Invalid start time. Please enter a date after {og_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}.")
                
                except ValueError:
                    print(f"\nError: Invalid format. Please use HH:MM:SS.ms.")

        elif(input_str == "2"):
            while True:
                time_str = input(f"The Original End time is {og_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\nEnter a time (HH:MM:SS.ms): ")
                try:
                    time_format = "%H:%M:%S.%f"
                    parsed_time_only = datetime.strptime(time_str, time_format).time()
                    new_parsed_time = datetime.combine(og_start_time.date(),parsed_time_only)
                    if (new_parsed_time >= og_start_time) and (new_parsed_time <= og_end_time) and (new_parsed_time > start_time):
                        print(f"\nSuccess! Created datetime object: {new_parsed_time}")
                        end_time = new_parsed_time
                        end_frame_index = index_from_time(og_start_time, og_end_time, end_time, len(og_intensity_time_data))
                        print(end_frame_index)
                        break
                    else:
                        print(f"Invalid start time. Please enter a date after {og_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}.")
                
                except ValueError:
                    print(f"\nError: Invalid format. Please use HH:MM:SS.ms.")

        elif(input_str == "3"):
            all_file_paths = load_folders(folder_path)
            vmin, vmax = plot_both_keograms(og_ns_keo_data[:,start_frame_index:end_frame_index], og_ew_keo_data[:,start_frame_index:end_frame_index], start_time, end_time, True)
            graph_images_from_frames(all_file_paths, start_frame_index, end_frame_index, start_time,end_time, vmin, vmax)

        elif(input_str == "4"):
            plot_both_keograms(og_ns_keo_data[:,start_frame_index*6:end_frame_index*6], og_ew_keo_data[:,start_frame_index*6:end_frame_index*6], start_time, end_time)
        
        elif(input_str == "5"):
            intensity_time_plot(og_intensity_time_data[start_frame_index:end_frame_index], start_time, end_time)
        
        elif(input_str == "6"):
            start_time = og_start_time
            end_time = og_end_time
            start_frame_index = 0
            end_frame_index = len(og_intensity_time_data)
        
        elif(input_str == "7"):
            break
        
        else:
            print("Invalid Input. Please try again")
    return og_ns_keo_data[:,start_frame_index*6:end_frame_index*6], og_ew_keo_data[:,start_frame_index*6:end_frame_index*6], og_intensity_time_data[start_frame_index:end_frame_index]


def main():

    # # Draw keogram North South ########################################################
    folder_path = "Big_Data\\Day_1\\ut02"
    keogram_data = run_parallel_folder(folder_path, get_col_6166)
    print("NS keogram data", keogram_data.shape)
    transposed_data = keogram_data.transpose(1, 0, 2)
    print("NS transposed data", transposed_data.shape)
    final_data = transposed_data.reshape(128, -1)
    print("NS final_data", final_data.shape)
    log_file = extract_text_from_log(folder_path)
    start_time_object, end_time_object = get_start_end_time(log_file)
    print(final_data.shape)
    plot_keogram(final_data, start_time_object, end_time_object)
    # #################################################################################

    # # Draw keogram East West ##########################################################
    # folder_path = "ut02"
    # keogram_data = run_parallel_folder(folder_path, get_row_6166)
    # print("EW keogram data", keogram_data.shape)
    # transposed_data = keogram_data.transpose(2, 0, 1)
    # print("EW transposed data", transposed_data.shape)
    # final_data = transposed_data.reshape(128, -1)
    # print("NS final_data", final_data.shape)
    # log_file = extract_text_from_log(folder_path)
    # start_time_object, end_time_object = get_start_end_time(log_file)
    # print(final_data.shape)
    # plot_keogram(final_data, start_time_object, end_time_object)
    # # #################################################################################   

    # # Create Time Plot
    # folder_path = "ut02"
    # intensity_time_data = run_parallel_folder(folder_path, get_row_col_6166_6166)
    # # intensity_time_data = run_parallel_folder(folder_path, get_row_col_6166_6166)
    # # test_data = intensity_time_data[:1000]
    # # print(intensity_time_data.shape)
    # log_file = extract_text_from_log(folder_path)
    # start_time_object, end_time_object = get_start_end_time(log_file)
    # # end_time_object = start_time_object+timedelta(seconds = 1000.0/56.7)
    # # print(start_time_object, end_time_object)
    # # intensity_time_plot(test_data, start_time_object, end_time_object)
    # intensity_time_plot(intensity_time_data, start_time_object, end_time_object)
    # intensity_time_plot(smooth_with_savgol(intensity_time_data, 11, 2), start_time_object, end_time_object)
    # intensity_time_plot(smooth_with_moving_average(intensity_time_data, 3), start_time_object, end_time_object)


    # # Analyzer
    # folder_path = "ut15"
    # data_analyzer(folder_path)




    # Get flash drive to copy 
    # sign up for oral presentation






if __name__ == "__main__":
    main()

