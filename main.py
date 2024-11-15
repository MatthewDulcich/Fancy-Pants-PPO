import numpy as np
import launch_ruffle
import enter_game
import multiprocessing
import time
import subprocess
import traceback
from selenium import webdriver
from selenium.webdriver.safari.options import Options
from PIL import Image
import mss

# Configuration
PORT = 8000
GAME_URL = f"http://localhost:{PORT}/launch_ruffle.html"
SAFARI_WEBDRIVER_URL = "http://localhost:4444"  # Safari’s WebDriver port

# Start the HTTP server for Ruffle Host
def start_ruffle_host():
    print("Starting the Ruffle host server...")
    server_process = subprocess.Popen(["python", "-m", "http.server", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for the server to start
    print("Ruffle host server started.")
    return server_process

# Start the Safari WebDriver and open the game URL
def start_safari_webdriver():
    print("Starting Safari WebDriver...")
    safari_process = subprocess.Popen(["safaridriver", "-p", "4444"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for WebDriver to start
    print("Safari WebDriver started.")

    # Initialize WebDriver and load the game URL
    safari_options = Options()
    safari_options.set_capability("browserName", "safari")
    driver = webdriver.Remote(command_executor=SAFARI_WEBDRIVER_URL, options=safari_options)
    print("Opening the game URL...")
    driver.get(GAME_URL)
    time.sleep(5)  # Wait for the page to load
    return safari_process, driver

# Fetch the canvas size and position from the game page
def fetch_canvas_position_and_size(driver):
    try:
        print("Attempting to access canvas through JavaScript...")
        canvas_info = driver.execute_script('''
            let rufflePlayer = document.querySelector('ruffle-player');
            if (rufflePlayer && rufflePlayer.shadowRoot) {
                let shadowRoot = rufflePlayer.shadowRoot;
                let canvas = shadowRoot.querySelector('canvas');
                if (canvas) {
                    let rect = canvas.getBoundingClientRect();
                    return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
                }
            }
            return null;
        ''')

        if canvas_info:
            print(f"Canvas Position and Size: Left = {canvas_info['left']}, "
                  f"Top = {canvas_info['top']}, Width = {canvas_info['width']}, Height = {canvas_info['height']}")
            return canvas_info
        else:
            print("Canvas element or window position was not found.")
            return None

    except Exception as e:
        print("An error occurred while fetching the canvas size.")
        print("Exception details:")
        traceback.print_exc()
        return None

# Fetch the offset for Safari's web content area
def fetch_content_offset(driver):
    offset = driver.execute_script('''
        return {
            xOffset: window.screenX + window.outerWidth - window.innerWidth,
            yOffset: window.screenY + window.outerHeight - window.innerHeight
        };
    ''')
    return offset

# Capture the observation space input (screenshot of the game area)
def capture_observation(canvas_info, content_offset):
    """
    Captures a screenshot of the specific game area dynamically based on the canvas info
    and starting from the top left of the web content in the Safari browser.
    """
    with mss.mss() as sct:
        # Adjust the monitor region for mss using content offset and canvas bounds
        monitor = {
            "top": int(content_offset['yOffset'] + canvas_info["top"]),
            "left": int(content_offset['xOffset'] + canvas_info["left"]),
            "width": int(canvas_info["width"]),
            "height": int(canvas_info["height"])
        }

        # Capture the region and convert to a numpy array
        screenshot = sct.grab(monitor)
        canvas_pixels = np.array(screenshot)
        return canvas_pixels

# Take 10 screenshots after game entry
def capture_multiple_observations(canvas_info, content_offset, num_screenshots=10):
    """
    Capture multiple screenshots to simulate the observation space for RL training.
    """
    print("Capturing screenshots of the game canvas region.")
    for i in range(num_screenshots):
        observation = capture_observation(canvas_info, content_offset)
        print(f"Screenshot {i+1} captured: shape = {observation.shape}")
        # save the image locally
        image = Image.fromarray(observation, 'RGBA')
        image.save(f"observation_{i+1}.png")

# Display a single screenshot
def display_canvas_image(canvas_pixels):
    image = Image.fromarray(canvas_pixels, 'RGBA')
    image.show()

# Handle initial game actions through enter_game
def run_enter_game():
    time.sleep(5)  # Wait for the server to start and the browser to open
    safari_window = enter_game.get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game.enter_game(safari_window)
    else:
        print("No Safari window found.")

# Cleanup function to ensure all processes are terminated
def cleanup(server_process, safari_process):
    print("Terminating the Ruffle host server and Safari WebDriver...")
    server_process.terminate()
    safari_process.terminate()
    server_process.wait()
    safari_process.wait()
    print("Processes terminated.")

# Main function orchestrating the steps
if __name__ == "__main__":
    # Kill any existing process on the port before starting
    launch_ruffle.kill_port(PORT)

    # Start the Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process, driver = start_safari_webdriver()

    try:
        # Start game automation process
        game_process = multiprocessing.Process(target=run_enter_game)
        game_process.start()
        game_process.join()  # Wait for `enter_game` to complete its actions

        # Fetch canvas size and position and Safari content offset
        canvas_info = fetch_canvas_position_and_size(driver)
        content_offset = fetch_content_offset(driver)

        if canvas_info:
            # Capture ten screenshots of the canvas region
            capture_multiple_observations(canvas_info, content_offset, num_screenshots=10)

    finally:
        # Cleanup to ensure no processes are left running
        cleanup(server_process, safari_process)

    print("End of main.py")