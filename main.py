import launch_ruffle
import enter_game
import multiprocessing
import time
import mss

# Define two capture modes with different regions
CAPTURE_REGIONS = {
    "mode_1": {"top": 195, "left": 260, "width": 808 - 260, "height": 597 - 197},
    "mode_2": {"top": 170, "left": 230, "width": 780 - 230, "height": 570 - 170}
}

def run_server():
    launch_ruffle.start_server()

def capture_game_screen(capture_mode="mode_1"):
    """
    Captures a screenshot of the specific game area based on the capture mode.
    """
    monitor = CAPTURE_REGIONS[capture_mode]
    with mss.mss() as sct:
        # Capture the region
        screenshot = sct.grab(monitor)
        return screenshot

def save_initial_screenshots(capture_mode="mode_1"):
    """
    Captures and saves ten initial screenshots based on the specified capture mode.
    """
    for i in range(10):
        # Capture the game screen based on selected mode
        screenshot = capture_game_screen(capture_mode=capture_mode)
        filename = f"screenshot_{i + 1}.png"
        
        # Save the screenshot
        mss.tools.to_png(screenshot.rgb, (screenshot.width, screenshot.height), output=filename)
        print(f"Saved {filename}")
        
        time.sleep(0.1)  # Brief delay between screenshots

def run_enter_game(capture_mode="mode_1"):
    # Wait for the server to start and the browser to open
    time.sleep(5)
    
    # Get the most recently opened Safari window
    safari_window = enter_game.get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game.enter_game(safari_window)
        
        # Capture and save screenshots based on the capture mode
        save_initial_screenshots(capture_mode=capture_mode)
        print("Initial screenshots captured successfully.")
    else:
        print("No Safari window found.")

if __name__ == "__main__":
    # Kill any existing process on the port before starting
    launch_ruffle.kill_port(launch_ruffle.PORT)
    
    # Define the capture mode (switch between "mode_1" and "mode_2" as needed)
    capture_mode = "mode_1"  # Set to "mode_2" for alternate dimensions
    
    # Create processes for the server and the enter_game function
    server_process = multiprocessing.Process(target=run_server)
    game_process = multiprocessing.Process(target=run_enter_game, args=(capture_mode,))
    
    # Start both processes
    server_process.start()
    game_process.start()
    
    # Wait for the game process to complete
    game_process.join()

    # Terminate the server process after game_process finishes
    server_process.terminate()
    server_process.join()  # Ensure it's cleaned up properly

    print("End of main.py")