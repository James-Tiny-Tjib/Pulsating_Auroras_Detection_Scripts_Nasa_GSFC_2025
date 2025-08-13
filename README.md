# **ROUGH DRAFT - SUBJECT TO CHANGE**


# Pulsating_Auroras_Detection_Scripts_Nasa_GSFC_2025
Space Club Scholar Intern Project 2025 - Detects and logs pulsating auroras from ground-based imager data of the sky.

**I highly recommend learning how to make a virtual environment so the downloaded packages don't interfere with other projects**

**Note: When copying file paths, add a "\" after every "\" to represent the backslash escape character. (E.g. "Data_Folder\ut02" -> "Data_Folder\\ut02"), or replace "\" with "/"**

# Example Input Data (Relavent Portions; the program should ignore any unnecessary files)
```
Hard Drive/ 
├── Day 1
│   ├── ut02 
|   |   ├── THA131208_02523649_16bit.log 
|   |   ├── THA131208_02523649_16bit.tif 
|   |   ├── THA131208_02523649_16bit_X2.tif 
|   |   ├── THA131208_02523649_16bit_X3.tif 
|   |   └── THA131208_02523649_16bit_X4.tif 
│   ├── ut03 
|   |   ├── THA131208_02523649_16bit.log 
|   |   ├── THA131208_02523649_16bit.tif 
|   |   ├── THA131208_02523649_16bit_X2.tif 
|   |   ├── THA131208_02523649_16bit_X3.tif 
|   |   └── THA131208_02523649_16bit_X4.tif 
│   └── ut04
|       ├── THA131208_02523649_16bit.log 
|       ├── THA131208_02523649_16bit.tif 
|       ├── THA131208_02523649_16bit_X2.tif 
|       ├── THA131208_02523649_16bit_X3.tif 
|       └── THA131208_02523649_16bit_X4.tif 
├── Day 2
│   ├── ut02 
|   |   ├── THA131208_02523649_16bit.log 
|   |   ├── THA131208_02523649_16bit.tif 
|   |   ├── THA131208_02523649_16bit_X2.tif 
|   |   ├── THA131208_02523649_16bit_X3.tif 
|   |   └── THA131208_02523649_16bit_X4.tif 
│   ├── ut03 
|   |   ├── THA131208_02523649_16bit.log 
|   |   ├── THA131208_02523649_16bit.tif 
|   |   ├── THA131208_02523649_16bit_X2.tif 
|   |   ├── THA131208_02523649_16bit_X3.tif 
|   |   └── THA131208_02523649_16bit_X4.tif 
│   └── ut04
|       ├── THA131208_02523649_16bit.log 
|       ├── THA131208_02523649_16bit.tif 
|       ├── THA131208_02523649_16bit_X2.tif 
|       ├── THA131208_02523649_16bit_X3.tif 
|       └── THA131208_02523649_16bit_X4.tif 
└── Day 3
    ├── ut02 
    |   ├── THA131208_02523649_16bit.log 
    |   ├── THA131208_02523649_16bit.tif 
    |   ├── THA131208_02523649_16bit_X2.tif 
    |   ├── THA131208_02523649_16bit_X3.tif 
    |   └── THA131208_02523649_16bit_X4.tif 
    ├── ut03
    |   ├── THA131208_02523649_16bit.log 
    |   ├── THA131208_02523649_16bit.tif 
    |   ├── THA131208_02523649_16bit_X2.tif  
    |   ├── THA131208_02523649_16bit_X3.tif 
    |   └── THA131208_02523649_16bit_X4.tif 
    └── ut04
        ├── THA131208_02523649_16bit.log 
        ├── THA131208_02523649_16bit.tif 
        ├── THA131208_02523649_16bit_X2.tif 
        ├── THA131208_02523649_16bit_X3.tif 
        └── THA131208_02523649_16bit_X4.tif 
```


# analyzer.py (deprecated)
Description: A stand-alone menu-based program that takes an hour of data (take the folder_path of an individual hour) and creates keograms, time-intensity plots, and allows the user to watch a video (and early implementation of a video player) given a start and end time. By setting a small interval , it can be useful for investigating details that are too small to see in the full-hour keogram. **This only works for the THA 2013-2014 data. Some values may need to change in order to use it for other datasets**<br>
Features:
* Commented-out blocks for individual full-hour keograms (North-South & East-West) and time_intensity_plots. To use this, replace the folder_path var with the full file_path of the data 
*  Analyzer Function: menu-based function that has the following options (enter 1-7):
  1. Edit Start Time (Enter in the form HH:MM:SS.ms e.g. 16:26:40.123)
  2. Edit End Time (Enter in the form HH:MM:SS.ms e.g 16:28:00.0)
  3. View Video (of set time interval, video player controls printed in console)
  4. View Keogram (Create a NS & EW keogram of set time interval)
  5. View Time Intensity Plot (of set time interval)
  6. Reset to Default (reset start & end times to original time intervals)
  7. Quit (Exit Program)

# full_day_keogram.py
Description: Given a Day of Data (Folder Path for 1 day), create a keogram for that entire day as a .png <br>
Variables:<br>
* RAM_FOR_LARGE_TIFF = 2 - Set how many GB's of RAM required for 1 TIFF file. This assigns how many workers are use for parallel processing limits how much RAM is being used.
* start_index, end_idex = 64,65 - Defines which row/column range to take from each frame
* increment_by = 10 - Skip that amount of frames whil making a keogram
* folder_path = "" - Set Day folder path to this variable
* direction = "NS" or "EW" - NS gets columns, "EW" gets rows
  
## `final_8_11_detection.py`

### Description
This script contains the core logic for detecting aurora pulsations. It can be used as a stand-alone program (for 1 hour of data) and also functions as a module that is automatically called by `final_8_11_master_analysis.py`.

### Functionality Overview
The script automates the process of pulsation detection through three stages:
1.  **Data Extraction**: Extracts average intensity time-series from regions in the image data.
2.  **Peak Detection**: Applies smoothing filters and algorithms to identify statistically significant peaks in the time-series data.
3.  **Vetting and Filtering**: Filters candidate peaks based on their properties (e.g., duration, sharpness) and merges nearby detections corresponding to the same event.

### Configurable Parameters (`analyze_pulsations_with_grid` function)
These parameters are hardcoded within the `analyze_pulsations_with_grid` function for standalone testing. When used as part of the suite, these values are overridden by the `CONFIG` dictionary in the master script.
* `FPS`: Frames per second of the source data.
* `NORMAL_PROMINENCE`: Main sensitivity for peak detection.
* `REDUCED_PROMINENCE`: Lowered sensitivity for peaks on bright backgrounds.
* `PIXEL_SIZE`: Side length of the analysis grid boxes.
* `HIGH_BASE_THRESHOLD`: Intensity level defining a "bright" background.
* `IMAGE_SIZE`: Dimensions of the image frames, e.g., `(128, 128)`.
* `GRID_DIMENSIONS`: Analysis grid layout, e.g., `(6, 6)`.
* `SIGMA_CLIP_THRESHOLD`: Value for robust outlier rejection.
* `MAX_PULSATION_SECONDS`: Maximum allowed duration of a pulsation.
* `MIN_PROMINENCE_WIDTH_RATIO`: Sharpness filter for pulsations.

## `final_8_11_detection_viewer.py`

### Description
This script provides the interactive video player for visualizing analysis results. It can be used as a stand-alone program (for 1 hour of data) and also functions like a module for `final_8_11_master_analysis.py`.

### Features
* **Full Movie Player**: Plays the entire video sequence with an overlay of analysis regions. The regions change color to highlight detected pulsation events in real-time.
* **Individual Pulsation Player**: Steps through each detected event one by one, playing a short clip for each.
* **Keyboard Controls**:
    * `Spacebar`: Pause or resume playback.
    * `r`: Rewind the video to the beginning.
    * `x`: Cycle through playback speeds.
    * `→` / `←`: Skip forward or backward (Full Movie Player only).

### Configurable Parameters (`if __name__ == '__main__'` block)
These parameters are available for standalone testing of the viewer.
* `DATA_FOLDER`: Path to the hourly data directory to be viewed.
* `REPORT_FILE`: Path to the `pulsation_report_grid.txt` file corresponding to the data.
* `BUFFER_SIZE`: Number of frames to hold in memory for smooth playback.
* `IMAGE_SIZE`: Dimensions of the image frames; must match the analysis script.
* `PIXEL_SIZE`: Size of the grid boxes; must match the analysis script.

## `final_8_11_detection_viewer_with_investigation.py`

### Description
This is an alternate version of the viewer that adds an **Investigation Mode**. This mode enables manual event marking during playback for a targeted follow-up analysis.

### How to Use
1.  **Enable Viewer**: To use this version, open `final_8_11_master_analysis.py` and set the `viewer_script_name` variable to `"final_8_11_detection_viewer_with_investigation.py"`.
2.  **Mark Events**: During playback in the Full Movie Player, **click inside any grid box** to mark that location and time as an event of interest. A confirmation will be printed to the console.
3.  **Run Analysis**: After closing the player window, a detailed analysis will run automatically on all marked events.

### Player Controls
Standard player controls (`Spacebar`, `r`, `x`, arrow keys) are identical to the standard viewer. The only added interaction is mouse-clicking to mark events.

### Outputs
Using the investigation mode generates new files:
* **`manual_analysis_plots/` folder**: A new directory containing detailed plots for each manually marked event.
* **`manual_analysis_log.txt`**: A text report summarizing the quantitative findings for all marked events.

## `final_8_11_master_analysis.py`

### Description
This script is the main controller for the Pulsation Analysis and Visualization Suite. It provides a menu to run a large-scale batch analysis on a dataset or to launch an interactive viewer for inspecting results.

### Operation
1.  **Parameter Configuration**: All operational parameters are located in the `CONFIG` dictionary at the top of the script. This section must be reviewed and edited before execution.
2.  **Execution**: The script is run from the command line: `python final_8_11_master_analysis.py`.
3.  **Mode Selection**: A menu will appear with two options:
    * **1. Run Batch Analysis**: Processes all data found in the `parent_data_directory`, generating reports and plots.
    * **2. Launch Interactive Viewer**: After an analysis is complete, this mode can be used to visually inspect the results for a specific hour of data.

### Tunable Variables (`CONFIG` Dictionary)
* `analysis_script_name`: The `.py` file containing the core detection logic.
* `viewer_script_name`: The `.py` file for the viewer. Change to `final_8_11_detection_viewer_with_investigation.py` to enable investigation mode.
* `parent_data_directory`: The file path to the root folder containing the day/hour data subdirectories.
* `main_output_folder`: The name of the directory where all analysis results will be saved.
* `keep_duplicates`: If `False`, nearby detections are merged into a single event. If `True`, all detections are kept.
* `generate_individual_plots`: If `True`, a detailed plot is created for each detected pulsation.
* `generate_full_region_plots`: If `True`, a summary plot of the full time-series is created for each grid region.
* `use_true_peaks`: If `True`, the peak intensity is determined from the absolute maximum in the raw data.
* `fps`: The frame rate (frames per second) of the source data.
* `grid_dimensions`: The number of rows and columns in the analysis grid (e.g., `(6, 6)`).
* `pixel_size`: The side length in pixels of the square analysis boxes.
* `sigma_clip_threshold`: The sigma value for outlier rejection during data extraction.
* `normal_prominence`: The primary sensitivity threshold for peak detection. Higher values detect more intense events.
* `reduced_prominence`: A lower sensitivity threshold for detecting peaks on an already bright background.
* `high_base_threshold`: The intensity level that defines a "bright" background.
* `max_pulsation_seconds`: The maximum allowed duration in seconds for a valid pulsation event.
* `min_prominence_width_ratio`: A sharpness filter (`prominence / width`) to exclude events that are not sufficiently "spiky".

### Outputs
* A main output folder containing a structured hierarchy of results by day and hour.
* For each processed hour, a text report and corresponding plot folders are generated.
* A `master_pulsation_report.txt` file is created, aggregating all detections from the entire analysis.



