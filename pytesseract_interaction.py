import time
import pyautogui
import pytesseract
import cv2
import numpy as np
from PIL import ImageGrab

# --- Helper Functions ---

def capture_region(region):
    """
    Captures a screenshot of the specified region and converts it to a grayscale numpy array.
    """
    screenshot = ImageGrab.grab(bbox=(
        region['left'],
        region['top'],
        region['left'] + region['width'],
        region['top'] + region['height']
    ))
    gray_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
    return gray_image


def extract_text_from_region(region):
    """
    Uses OCR to extract text from a specified screen region.
    """
    gray_image = capture_region(region)
    detected_text = pytesseract.image_to_string(gray_image).strip()
    return detected_text


# --- OCR and Interaction Functions ---

def get_tab_bar_region(safari_window):
    """
    Computes the region of the horizontal tab bar within the Safari window.
    """
    safari_top = safari_window['top'] + 50  # Add top bar offset
    safari_left = safari_window['left']
    safari_width = safari_window['width']

    # Assume the tab bar is a fixed height
    tab_bar_height = 40  # Adjust based on macOS Safari's tab bar height

    # Define the region for the tab bar
    tab_bar_region = {
        'left': safari_left,
        'top': safari_top,
        'width': safari_width,
        'height': tab_bar_height
    }

    print("Tab Bar Region:", tab_bar_region)
    return tab_bar_region


def handle_reload_bar(tab_bar_region):
    """
    Detects and dismisses the 'This webpage was reloaded' bar using the tab bar region.
    """
    detected_text = extract_text_from_region(tab_bar_region)
    print("Detected Text in Reload Bar:", detected_text)

    if "This webpage was reloaded" in detected_text:
        print("Reload bar detected. Attempting to dismiss it...")
        # Click the "X" button
        x_button_coords = (
            tab_bar_region['left'] + 17, 
            tab_bar_region['top'] + 17
        )
        pyautogui.moveTo(x_button_coords[0], x_button_coords[1], duration=0.2)
        pyautogui.click()
        time.sleep(1)  # Allow the reload bar to disappear
        print("Dismissed the reload bar.")
        return True
    else:
        print("Reload bar not detected.")
        return False

def wait_for_play_now_text(region, target_text, timeout=60, check_interval=1):
    """
    Waits until the specified text appears in a given screen region.
    """
    start_time = time.time()

    iteration = 0
    while time.time() - start_time < timeout:
        detected_text = extract_text_from_region(region)
        print(f"Iteration {iteration} / {timeout}: Detected Text: {detected_text}")

        if target_text.lower() in detected_text.lower():
            return True

        time.sleep(check_interval)  # Wait before checking again
        iteration += 1

    return False  # Timeout reached

def click_center_of_region(region):
    """
    Clicks the center of the given screen region.
    
    :param region: Dictionary containing the region with keys 'left', 'top', 'width', and 'height'.
    """
    if not all(key in region for key in ['left', 'top', 'width', 'height']):
        raise ValueError("Region dictionary must contain 'left', 'top', 'width', and 'height' keys.")

    center_x = region['left'] + region['width'] // 2
    center_y = region['top'] + region['height'] // 2
    pyautogui.moveTo(center_x, center_y, duration=0.1)
    pyautogui.click()

# --- Main Script ---

if __name__ == "__main__":
    # Example Safari window data (replace this with real fetch logic)
    safari_window = {
        'top': 100,
        'left': 200,
        'width': 800,
        'height': 600
    }

    # Define the region to monitor for text
    button_region = (200, 300, 140, 40)  # Relative coordinates (x, y, width, height)

    # Adjust the button region for top and bottom bars
    top_bar_offset = 50
    bottom_bar_offset = 30
    button_region_adjusted = (
        button_region[0],
        button_region[1] + top_bar_offset,
        button_region[2],
        button_region[3]
    )
    print("Adjusted Button Region:", button_region_adjusted)

    # Handle reload bar
    tab_bar_region = get_tab_bar_region(safari_window)
    handle_reload_bar(tab_bar_region)

    # Wait for the "Play Again" text and click
    if wait_for_play_now_textbar(region=button_region_adjusted, target_text="Play Again"):
        print("Detected 'Play Again'. Clicking the button...")
        click_center_of_region(button_region_adjusted)
    else:
        print("Timeout reached. 'Play Again' not detected.")