import time
import pyautogui
import pytesseract

# --- OCR and Interaction Functions ---

def wait_for_text(region, target_text, timeout=60, check_interval=1):
    """
    Waits until the specified text appears in a given screen region.
    """
    start_time = time.time()

    iteration = 0
    while time.time() - start_time < timeout:
        screenshot = pyautogui.screenshot(region=region)
        gray_image = screenshot.convert('L')  # Convert to grayscale for better OCR
        detected_text = pytesseract.image_to_string(gray_image).strip()
        print(f"Iteration {iteration} / {timeout}: Detected Text: {detected_text}")

        if target_text.lower() in detected_text.lower():
            return True

        time.sleep(check_interval)  # Wait before checking again
        iteration += 1

    return False  # Timeout reached


def click_center_of_region(region):
    """
    Clicks the center of the given screen region.
    """
    center_x = region[0] + region[2] // 2
    center_y = region[1] + region[3] // 2
    pyautogui.moveTo(center_x, center_y, duration=0.1)
    pyautogui.click()

# --- Main Script ---

if __name__ == "__main__":
    # Define the region to monitor for text
    button_region = (200, 300, 140, 40)  # Relative coordinates (x, y, width, height)

    # Wait for the "Play Again" text and click
    if wait_for_text(region=button_region, target_text="Play Again"):
        print("Detected 'Play Again'. Clicking the button...")
        click_center_of_region(button_region)
    else:
        print("Timeout reached. 'Play Again' not detected.")