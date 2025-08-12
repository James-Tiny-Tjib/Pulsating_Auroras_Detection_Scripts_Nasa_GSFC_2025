# Pulsating_Auroras_Detection_Scripts_Nasa_GSFC_2025
Space Club Scholar Intern Project 2025 - Detects and logs pulsating auroras from ground-based imager data of the sky.

**Note: When copying file paths, add a "\" after every "\" to represent the backslash escape character. (E.g. "Data_Folder\ut02" -> "Data_Folder\\ut02")**

# Input Data


# analyzer.py (deprecated)
Description: A stand-alone menu-based program that takes an hour of data and creates keograms, time-intensity plots, and allows the user to watch a video ( and early implementation of a video player) given a start and end time. By setting a small interval , it can be useful for investigating details that are too small to see in the full-hour keogram. **This only works for the THA 2013-2014 data. You may need to change some values in order to use it for other datasets**<br>

Features:
* Commented-out blocks for individual full-hour keograms (North-South & East-West) and time_intensity_plots. To use this, replace the folder_path var with the full file_path of the data you want to use
*  Analyzer Function: menu-based function that has the following options (enter 1-7):
  1. Edit Start Time (Enter in the form HH:MM:SS.ms e.g. 16:26:40.123)
  2. Edit End Time (Enter in the form HH:MM:SS.ms e.g 16:28:00.0)
  3. View Video (of set time interval, video player controls printed in console)
  4. View Keogram (Create a NS & EW keogram of set time interval)
  5. View Time Intensity Plot (of set time interval)
  6. Reset to Default (reset start & end times to original time intervals)
  7. Quit (Exit Program)

