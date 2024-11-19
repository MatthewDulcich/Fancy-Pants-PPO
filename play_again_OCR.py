import time
import pyautogui
import pytesseract

# --- OCR and Interaction Functions ---

def wait_for_text(region, target_text, timeout=30, check_interval=1):
    """
    Waits until the specified text appears in a given screen region.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        screenshot = pyautogui.screenshot(region=region)
        gray_image = screenshot.convert('L')  # Convert to grayscale for better OCR
        detected_text = pytesseract.image_to_string(gray_image).strip()
        print(f"Detected Text: {detected_text}")

        if target_text.lower() in detected_text.lower():
            return True

        time.sleep(check_interval)  # Wait before checking again

    return False  # Timeout reached


def click_center_of_region(region):
    """
    Clicks the center of the given screen region.
    """
    center_x = region[0] + region[2] // 2
    center_y = region[1] + region[3] // 2
    pyautogui.moveTo(center_x, center_y, duration=0.5)
    pyautogui.click()