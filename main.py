import launch_ruffle
import enter_game
import multiprocessing
import time

def run_server():
    launch_ruffle.start_server()

def run_enter_game():
    # Wait for the server to start and the browser to open
    time.sleep(5)
    
    # Get the most recently opened Safari window
    safari_window = enter_game.get_most_recent_window_by_owner("Safari")
    if safari_window:
        enter_game.enter_game(safari_window)
    else:
        print("No Safari window found.")

if __name__ == "__main__":
    # Kill any existing process on the port before starting
    launch_ruffle.kill_port(launch_ruffle.PORT)
    
    # Create processes for the server and the enter_game function
    server_process = multiprocessing.Process(target=run_server)
    game_process = multiprocessing.Process(target=run_enter_game)
    
    # Start both processes
    server_process.start()
    game_process.start()
    
    # Wait for both processes to complete
    server_process.join()
    game_process.join()