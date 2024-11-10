import subprocess
import time
import os
import signal
from selenium import webdriver
from selenium.webdriver.safari.options import Options
import traceback

# Configuration
PORT = 8000
GAME_URL = f"http://localhost:{PORT}/launch_ruffle.html"
SAFARI_WEBDRIVER_URL = "http://localhost:4444"  # Safari’s WebDriver port

# Function to start the HTTP server for Ruffle Host
def start_ruffle_host():
    print("Starting the Ruffle host server...")
    server_process = subprocess.Popen(["python", "-m", "http.server", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for the server to start
    print("Ruffle host server started.")
    return server_process

# Function to start the Safari WebDriver
def start_safari_webdriver():
    print("Starting Safari WebDriver...")
    safari_process = subprocess.Popen(["safaridriver", "-p", "4444"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for WebDriver to start
    print("Safari WebDriver started.")
    return safari_process

# Function to fetch the canvas size
def fetch_canvas_size():
    # Initialize the Safari WebDriver with options
    safari_options = Options()
    safari_options.set_capability("browserName", "safari")
    driver = webdriver.Remote(command_executor=SAFARI_WEBDRIVER_URL, options=safari_options)
    
    try:
        print("Opening the game URL...")
        driver.get(GAME_URL)
        time.sleep(5)  # Wait for full page load

        print("Attempting to access canvas through JavaScript...")

        # JavaScript to navigate through shadow DOM and access the canvas element
        canvas_element = driver.execute_script('''
            let rufflePlayer = document.querySelector('ruffle-player');
            if (rufflePlayer && rufflePlayer.shadowRoot) {
                let shadowRoot = rufflePlayer.shadowRoot;
                let canvas = shadowRoot.querySelector('canvas');
                return canvas;
            } else {
                return null;
            }
        ''')

        if canvas_element:
            print("Canvas element found. Fetching its size...")
            canvas_width = driver.execute_script("return arguments[0].width;", canvas_element)
            canvas_height = driver.execute_script("return arguments[0].height;", canvas_element)
            print(f"Canvas Size: Width = {canvas_width}, Height = {canvas_height}")
        else:
            print("Canvas element was not found in shadow DOM.")

    except Exception as e:
        print("An error occurred while fetching the canvas size.")
        print("Exception details:")
        traceback.print_exc()

    finally:
        print("Closing the driver.")
        driver.quit()

# Main function to orchestrate the entire process
def main():
    # Start the Ruffle host server
    server_process = start_ruffle_host()
    
    # Start Safari WebDriver
    safari_process = start_safari_webdriver()

    # Fetch the canvas size
    fetch_canvas_size()

    # Cleanup: Terminate the processes
    print("Terminating the Ruffle host server and Safari WebDriver...")
    server_process.terminate()
    safari_process.terminate()
    print("Processes terminated.")

if __name__ == "__main__":
    main()
