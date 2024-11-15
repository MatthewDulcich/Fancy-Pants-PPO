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

# Environment setup functions
def start_ruffle_host():
    """
    Starts the HTTP server for hosting the Ruffle game and returns the process.
    """
    print("Starting the Ruffle host server...")
    server_process = subprocess.Popen(["python", "-m", "http.server", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    print("Ruffle host server started.")
    return server_process

def start_safari_webdriver():
    """
    Starts Safari WebDriver and opens the game URL, returning the process and WebDriver.
    """
    print("Starting Safari WebDriver...")
    safari_process = subprocess.Popen(["safaridriver", "-p", "4444"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    print("Safari WebDriver started.")

    safari_options = Options()
    safari_options.set_capability("browserName", "safari")
    driver = webdriver.Remote(command_executor=SAFARI_WEBDRIVER_URL, options=safari_options)
    print("Opening the game URL...")
    driver.get(GAME_URL)
    time.sleep(5)
    return safari_process, driver

# Game setup and automation functions
def run_enter_game():
    """
    Automates the initial game entry actions, such as navigating past menus or starting the game.
    """
    time.sleep(5)
    safari_window = enter_game.get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game.enter_game(safari_window)
    else:
        print("No Safari window found.")

# Observation and canvas functions
def fetch_canvas_position_and_size(driver):
    """
    Fetches the canvas size and position for targeted screen capture within the game area.
    """
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
            print(f"Canvas Position and Size: Left = {canvas_info['left']}, Top = {canvas_info['top']}, Width = {canvas_info['width']}, Height = {canvas_info['height']}")
            return canvas_info
        else:
            print("Canvas element or window position was not found.")
            return None
    except Exception:
        print("An error occurred while fetching the canvas size.")
        traceback.print_exc()
        return None

def fetch_content_offset(driver):
    """
    Retrieves the offset of the web content area to exclude the browser's menu and toolbars.
    """
    return driver.execute_script('''
        return {
            xOffset: window.screenX + window.outerWidth - window.innerWidth,
            yOffset: window.screenY + window.outerHeight - window.innerHeight
        };
    ''')

def capture_observation(canvas_info, content_offset):
    """
    Captures a screenshot of the specific game area dynamically based on the canvas info
    and starting from the top-left of the web content in the Safari browser.
    """
    with mss.mss() as sct:
        monitor = {
            "top": int(content_offset['yOffset'] + canvas_info["top"]),
            "left": int(content_offset['xOffset'] + canvas_info["left"]),
            "width": int(canvas_info["width"]),
            "height": int(canvas_info["height"])
        }
        screenshot = sct.grab(monitor)
        canvas_pixels = np.array(screenshot)
        return canvas_pixels

def capture_multiple_observations(canvas_info, content_offset, num_screenshots=10):
    """
    Captures multiple screenshots to simulate the observation space for RL training.
    """
    print("Capturing screenshots of the game canvas region.")
    for i in range(num_screenshots):
        observation = capture_observation(canvas_info, content_offset)
        print(f"Screenshot {i+1} captured: shape = {observation.shape}")
        # Placeholder for PPO model integration point: this is where observations would be passed to the model
        # save_observation(observation, i)  # Uncomment if you wish to save observations locally
    print(observation[:10, :10, 0])  # Sample pixel data for verification

# Helper functions for saving/displaying screenshots
def display_canvas_image(canvas_pixels):
    image = Image.fromarray(canvas_pixels, 'RGBA')
    image.show()

# Cleanup function
def cleanup(server_process, safari_process):
    """
    Ensures all processes are terminated properly.
    """
    print("Terminating the Ruffle host server and Safari WebDriver...")
    server_process.terminate()
    safari_process.terminate()
    server_process.wait()
    safari_process.wait()
    print("Processes terminated.")

# Main function with modular steps conducive to PPO integration
def main():
    # Kill any existing process on the port before starting
    launch_ruffle.kill_port(PORT)

    # Start environment: Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process, driver = start_safari_webdriver()

    try:
        # Game setup: automate initial game actions
        game_process = multiprocessing.Process(target=run_enter_game)
        game_process.start()
        game_process.join()

        # Observation setup: fetch canvas size and content offset
        canvas_info = fetch_canvas_position_and_size(driver)
        content_offset = fetch_content_offset(driver)

        if canvas_info:
            # Observation collection
            capture_multiple_observations(canvas_info, content_offset, num_screenshots=10)

    finally:
        # Cleanup to terminate processes
        cleanup(server_process, safari_process)

    print("End of main.py")

if __name__ == "__main__":
    main()