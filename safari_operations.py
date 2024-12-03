import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import config_handler

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
        print("Safari Window Coordinates:", coordinates)
        return {
            "left": coordinates[0],
            "top": coordinates[1],
            "width": coordinates[2] - coordinates[0],
            "height": coordinates[3] - coordinates[1],
        }
    except Exception as e:
        print("Error fetching Safari window coordinates:", e)
        return None

def adjust_for_menu_bar(safari_window, region):
    """
    Adjusts the button region coordinates based on Safari's window position
    and the menu bar offset.
    """
    print(f"Safari Window Coordinates: {safari_window}")
    safari_left = safari_window['left']
    safari_top = safari_window['top']
    menu_bar_offset = 88  # Adjust based on macOS menu bar size

    return (
        safari_left + region[0],
        safari_top + menu_bar_offset + region[1],
        region[2],
        region[3],
    )

def get_canvas_position_selenium():
    """
    Fetches the canvas element's position and size dynamically using Selenium.
    """
    driver = None
    try:
        # Initialize Safari WebDriver
        driver = webdriver.Safari()

        # Open the game URL (replace with your actual game URL)
        driver.get(config_handler.get_config("GAME_URL"))

        # Wait for the page to load (adjust the delay if needed)
        time.sleep(5)

        # Locate the <canvas> element
        canvas = driver.find_element(By.TAG_NAME, "canvas")
        if not canvas:
            raise ValueError("Canvas element not found")

        # Get the canvas position and size
        canvas_rect = canvas.rect
        canvas_position = {
            "left": canvas_rect["x"],
            "top": canvas_rect["y"],
            "width": canvas_rect["width"],
            "height": canvas_rect["height"],
        }

        print("Canvas Position (Selenium):", canvas_position)
        return canvas_position

    except Exception as e:
        print("Error fetching canvas position with Selenium:", e)
        return None

    finally:
        # Quit the driver only if it was initialized
        if driver:
            driver.quit()

# --- Main Script ---

if __name__ == "__main__":
    # Get Safari window coordinates
    safari_window = get_safari_window_coordinates()
    if safari_window:
        # Get canvas position from DOM
        canvas_position = get_canvas_position_selenium()
        if canvas_position:
            # Adjust button region for Safari window
            button_region = (canvas_position["left"], canvas_position["top"], canvas_position["width"], canvas_position["height"])
            adjusted_button_region = adjust_for_menu_bar(safari_window, button_region)
            print("'Play Now!' Button Region (relative):", button_region)
            print("Adjusted Button Region (global):", adjusted_button_region)
        else:
            print("Failed to fetch canvas position from DOM.")
    else:
        print("Failed to fetch Safari window coordinates.")
