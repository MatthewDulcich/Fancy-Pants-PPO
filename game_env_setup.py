import subprocess
import time
import mss
import numpy as np

# --- Game Environment Functions ---

# Configuration
PORT = 8000
GAME_URL = f"http://localhost:{PORT}/launch_ruffle.html"
SAFARI_WEBDRIVER_URL = "http://localhost:4444"

def start_ruffle_host(port=8000):
    """
    Starts the HTTP server for hosting the Ruffle game and returns the process.
    """
    print("Starting the Ruffle host server...")
    server_process = subprocess.Popen(
        ["python", "-m", "http.server", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(3)
    print("Ruffle host server started.")
    return server_process


def start_safari_webdriver(game_url):
    """
    Starts Safari WebDriver and opens the game URL, returning the process and WebDriver.
    """
    from selenium import webdriver
    from selenium.webdriver.safari.options import Options

    print("Starting Safari WebDriver...")
    safari_process = subprocess.Popen(["safaridriver", "-p", "4444"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    print("Safari WebDriver started.")

    safari_options = Options()
    safari_options.set_capability("browserName", "safari")
    driver = webdriver.Remote(command_executor="http://localhost:4444", options=safari_options)
    print("Opening the game URL...")
    driver.get(game_url)
    time.sleep(5)
    return safari_process, driver


def fetch_canvas_position_and_size(driver):
    """
    Fetches the canvas size and position for targeted screen capture within the game area.
    """
    try:
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
    except Exception as e:
        print("An error occurred while fetching the canvas size.")
        print(e)
        return None


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
            "height": int(canvas_info["height"]),
        }
        screenshot = sct.grab(monitor)
        canvas_pixels = np.array(screenshot)
        print("Screenshot captured: shape =", canvas_pixels.shape)
        return canvas_pixels


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

# --- Main Script ---

if __name__ == "__main__":
    # Start the Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process, driver = start_safari_webdriver(GAME_URL)

    # Fetch canvas position and size
    canvas_info = fetch_canvas_position_and_size(driver)
    if not canvas_info:
        cleanup(server_process, safari_process)
        exit()

    # Fetch content offset to exclude browser menu and toolbars
    content_offset = driver.execute_script('''
        return {
            xOffset: window.screenX + window.outerWidth - window.innerWidth,
            yOffset: window.screenY + window.outerHeight - window.innerHeight
        };
    ''')

    # Capture the game observation
    canvas_pixels = capture_observation(canvas_info, content_offset)

    # Cleanup processes
    cleanup(server_process, safari_process)