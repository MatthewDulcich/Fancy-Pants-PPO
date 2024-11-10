import os
import signal
import subprocess
import http.server
import socketserver
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Port and directory configuration
PORT = 8000
DIRECTORY = "."  # Update this with your folder path

# Function to kill the process on the specified port
def kill_port(port):
    try:
        # Find the process ID using the port
        result = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pid = int(result.strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Process on port {port} terminated.")
    except subprocess.CalledProcessError:
        print(f"No process is running on port {port}.")

# Kill any existing process on port 8000 before starting the server
kill_port(PORT)

# Start the server
os.chdir(DIRECTORY)
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    # Open the game page in Safari browser
    server_url = f"http://localhost:{PORT}/launch_ruffle.html"
    print("Serving at port", PORT)
    
    # Start the Selenium WebDriver for Safari with the remote address
    safari_options = webdriver.SafariOptions()
    safari_options.add_argument("--remote-allow-origins=*")
    driver = webdriver.Safari(options=safari_options)
    
    # Open the game page with Selenium
    driver.get(server_url)
    print("Page opened in Safari browser.")

    # Wait for the page to load
    time.sleep(5)  # Adjust the sleep time if necessary

    # Use Selenium to find the container and canvas elements
    try:
        # Locate the container element by ID
        container = driver.find_element(By.ID, "container")
        print("Container element found.")

        # Locate the canvas within the container
        canvas = container.find_element(By.TAG_NAME, "canvas")
        width = canvas.get_attribute("width")
        height = canvas.get_attribute("height")
        print(f"Canvas element found with dimensions: {width} x {height}")
        
    except Exception as e:
        print(f"Error locating elements: {e}")
    finally:
        input("Press Enter to stop the server and close the browser...")

        # Stop the server and close the browser
        driver.quit()
        print("Browser closed.")
        httpd.shutdown()
        print("Server stopped.")
