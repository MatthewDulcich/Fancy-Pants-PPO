from game_env_setup import start_ruffle_host, launch_safari_host, cleanup
import config_handler as config_handler
import json

# Load configuration
config = config_handler.load_config("game_config.json")

# function to kill the port
def kill_port(port):
    """
    Kills the process running on the specified port if it exists.
    """
    import os
    import subprocess

    try:
        # Check for a process running on the port
        output = subprocess.check_output(["lsof", "-t", "-i", f":{port}"])
        pid = int(output.decode().strip())
        os.system(f"kill {pid}")
        print(f"Process running on port {port} terminated successfully.")
    except subprocess.CalledProcessError:
        # No process found on the port
        print(f"No process is running on port {port}.")
    except Exception as e:
        print(f"An error occurred while trying to terminate the process on port {port}:", e)

if __name__ == "__main__":
    # Start the Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process = launch_safari_host(config['GAME_URL'])

    # If keyboard Interrupt occurs, cleanup the processes
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print(" Keyboard Interrupt detected. Cleaning up...")
        cleanup(server_process, safari_process)
        