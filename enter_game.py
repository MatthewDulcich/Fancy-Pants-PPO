import pyautogui
import time
from window_management import get_most_recent_window_by_owner
from safari_operations import get_safari_window_coordinates, adjust_for_menu_bar
from pytesseract_interaction import wait_for_play_now_text, click_center_of_region, get_tab_bar_region, handle_reload_bar
import config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

# --- Game Automation Functions ---

def enter_game(window, pre_loaded=False):
    """
    Automates entering the game by dynamically detecting the "Play Now" button
    and pressing the required keys to start the game.
    """
    # Step 1: Get window position and size
    window_bounds = window['kCGWindowBounds']
    window_left = window_bounds['X']
    window_top = window_bounds['Y']

    # Initial click to focus the game window
    pyautogui.moveTo(window_left + 300, window_top + 300, duration=0.1)  # Center of initial area
    pyautogui.click()

    # Step 2: Get Safari window coordinates
    safari_window = get_safari_window_coordinates()
    if not safari_window:
        print("Failed to fetch Safari window coordinates.")
        return
    
    # Add offset which sometimes triggers
    if config.get('tabs_present', False):
        tab_offset = 25
    else:
        tab_offset = 0

    # Page reload check (inserted between Step 2 and Step 3)
    tab_bar_region = get_tab_bar_region(safari_window, offset = tab_offset)
    if handle_reload_bar(tab_bar_region):
        print("Handled reload bar. Proceeding to click play again")

    # Step 3: Define button region and adjust for Safari window position
    play_again_button_region = {"left": 205, "top": 300 + tab_offset, "width": 140, "height": 40}  # Relative coordinates (x, y, width, height)
    adjusted_button_region = adjust_for_menu_bar(safari_window, play_again_button_region)
    # print("Adjusted Button Region:", adjusted_button_region)

    # Initial click to focus the game window
    pyautogui.moveTo(window_left + 300, window_top + 300, duration=0.1)  # Center of initial area
    pyautogui.click()

    # Step 4: Wait for the "Play Now" text and click
    if wait_for_play_now_text(region=adjusted_button_region, target_text="Play Now"):
        print("Detected 'Play Now'. Clicking the button...")
        click_center_of_region(adjusted_button_region)
    else:
        print("Timeout reached. 'Play Now' not detected.")
        return

    if not pre_loaded:
        time.sleep(13)  # Wait for the game to load
        print("Loading steps for new game...")
        # Step 5: Automate key presses to navigate into the game
        key_sequence = ['up', 'up', 'up', 's']  # Navigation keys to enter the game
        for key in key_sequence:
            time.sleep(2)
            pyautogui.press(key)
            print(f"Pressing {key}")

        # Step 6: Simulate arrow key movements
        pyautogui.keyDown('left')
        pyautogui.keyDown('up')
        time.sleep(1)
        pyautogui.keyUp('left')
        pyautogui.keyUp('up')
        time.sleep(3)
        pyautogui.keyDown('right')
        pyautogui.keyDown('up')
    
    else:
        time.sleep(5)
        print("Loading steps for pre-loaded game...")
        pyautogui.keyDown('s')
        pyautogui.keyDown('left')
        pyautogui.keyDown('up')
        print("Pressing 's' to jump out of bed...")
        time.sleep(2)
        print("Exiting the room...")
        time.sleep(5)
        print("Entering the tutorial level...")
        pyautogui.keyUp('s')
        pyautogui.keyUp('left')
        pyautogui.keyUp('up')
        time.sleep(10)
        pyautogui.keyDown('right')
        time.sleep(4)
        pyautogui.keyDown('s')
        time.sleep(2)
        pyautogui.keyUp('right')
        pyautogui.keyUp('s')
        time.sleep(2)
      

    print("Done entering game. Training begins now!")


# --- Main Script ---

if __name__ == "__main__":
    # Get the most recently opened Safari window
    safari_window = get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game(safari_window)
    else:
        print("No Safari window found.")