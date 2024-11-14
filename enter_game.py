import pyautogui
import time
import Quartz
from AppKit import NSWorkspace

# This function retrieves a list of all windows with their titles and owner names
def get_window_list():
    window_list = []
    window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    for window_info in window_info_list:
        window_title = window_info.get('kCGWindowName', 'No Title')
        window_owner_name = window_info.get('kCGWindowOwnerName', 'Unknown')
        window_layer = window_info.get('kCGWindowLayer', 0)
        window_list.append((window_title, window_owner_name, window_layer, window_info))
    return window_list

# This function will look for the most recently opened window with a specific owner name
def get_most_recent_window_by_owner(owner_name):
    windows = get_window_list()
    # Filter windows by owner name
    owner_windows = [window for window in windows if owner_name.lower() in window[1].lower()]
    # Sort windows by layer (higher layer means more recent)
    owner_windows.sort(key=lambda x: x[2], reverse=True)
    return owner_windows[0][3] if owner_windows else None

def list_windows():
    windows = get_window_list()
    # Sort windows by owner name
    windows.sort(key=lambda x: x[1])
    print("Open windows:")
    for i, (title, owner, _, _) in enumerate(windows):
        print(f"{i + 1}: {title} (Owner: {owner})")
    return windows

def enter_game(window):
    # Get the window's position and size
    window_bounds = window['kCGWindowBounds']
    window_left = window_bounds['X']
    window_top = window_bounds['Y']
    window_width = window_bounds['Width']
    window_height = window_bounds['Height']
    
    # Calculate the top-right corner coordinates
    top_right_x = window_left + 300
    top_right_y = window_top + 300
    
    # Move the mouse to the top-right corner of the window
    pyautogui.moveTo(top_right_x, top_right_y, duration=0.5)
    pyautogui.click()
    
    # Send keyboard inputs
    # pyautogui.press('enter')

    time.sleep(10)

    # Move the mouse to the top-right corner of the window
    pyautogui.moveTo(top_right_x, top_right_y + 198, duration=0.5)
    pyautogui.click()

    time.sleep(10)
    pyautogui.press('up')

    time.sleep(3)
    pyautogui.press('up')

    time.sleep(3)
    pyautogui.press('up')

    time.sleep(10)
    pyautogui.press('s')
    print("Pressing s")

    # Press and hold the left arrow key for 1 second
    pyautogui.keyDown('left')
    pyautogui.keyDown('up')
    time.sleep(1)
    pyautogui.keyUp('up')
    pyautogui.keyUp('left')

    time.sleep(10)

    print('done with enter_game')


if __name__ == "__main__":
    windows = list_windows()
    # print(windows)
    # Get the most recently opened Safari window
    safari_window = get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game(safari_window)
    else:
        print("No Safari window found.")