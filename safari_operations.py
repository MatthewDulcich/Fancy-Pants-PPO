import subprocess

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
    try:
        output = subprocess.check_output(["osascript", "-e", script])
        coordinates = [int(coord) for coord in output.decode().strip().split(", ")]
        return {
            "left": coordinates[0],
            "top": coordinates[1],
            "width": coordinates[2] - coordinates[0],
            "height": coordinates[3] - coordinates[1],
        }
    except Exception as e:
        print("Error fetching Safari window coordinates:", e)
        return None


def adjust_for_menu_bar(safari_window, button_region):
    """
    Adjusts the button region coordinates based on Safari's window position
    and the menu bar offset.
    """
    safari_left = safari_window["left"]
    safari_top = safari_window["top"]
    menu_bar_offset = 88  # Adjust based on macOS menu bar size

    return (
        safari_left + button_region[0],  # Adjust for Safari window's left
        safari_top + menu_bar_offset + button_region[1],  # Adjust for Safari window's top and menu bar
        button_region[2],  # Width stays the same
        button_region[3],  # Height stays the same
    )
