import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Helper Functions ---

def run_applescript(script):
    """
    Runs an AppleScript command and returns its output.
    """
    try:
        output = subprocess.check_output(["osascript", "-e", script], universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"AppleScript error: {e.output}")
        return None


def parse_coordinates(output):
    """
    Parses the coordinates output from AppleScript.
    """
    try:
        coordinates = [int(coord) for coord in output.split(", ")]
        # logging.info(f"Parsed Coordinates: {coordinates}")
        return {
            "left": coordinates[0],
            "top": coordinates[1],
            "width": coordinates[2] - coordinates[0],
            "height": coordinates[3] - coordinates[1],
        }
    except (ValueError, IndexError):
        logging.error(f"Failed to parse coordinates: {output}")
        return None


# --- Safari-Specific Functions ---

def get_safari_window_coordinates():
    """
    Fetches Safari's active window position and size using AppleScript.
    """
    script = '''
    tell application "Safari"
        set safariBounds to bounds of front window
    end tell
    return safariBounds
    '''
    output = run_applescript(script)
    return parse_coordinates(output) if output else None


def adjust_for_menu_bar(safari_window, region, menu_bar_offset=88):
    """
    Adjusts the button region coordinates based on Safari's window position
    and the menu bar offset.
    """
    if not safari_window or not region:
        logging.error("Invalid input: safari_window or region is None.")
        return None

    # logging.info(f"Safari Window Coordinates: {safari_window}")
    safari_left, safari_top = safari_window['left'], safari_window['top']

    return {
        "left": safari_left + region["left"],
        "top": safari_top + menu_bar_offset + region["top"],
        "width": region["width"],
        "height": region["height"],
    }

# --- Main Script ---

if __name__ == "__main__":
    # Fetch Safari window coordinates
    safari_window = get_safari_window_coordinates()
    if safari_window:
        # Define a button region (example values, replace with actual logic as needed)
        button_region = {"left": 200, "top": 300, "width": 140, "height": 40}

        # Adjust button region for Safari window
        adjusted_button_region = adjust_for_menu_bar(safari_window, button_region)
        if adjusted_button_region:
            logging.info(f"Button Region (relative): {button_region}")
            logging.info(f"Adjusted Button Region (global): {adjusted_button_region}")
        else:
            logging.error("Failed to adjust button region.")
    else:
        logging.error("Failed to fetch Safari window coordinates.")