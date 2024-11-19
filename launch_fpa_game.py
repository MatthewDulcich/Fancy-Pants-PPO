from game_env_setup import start_ruffle_host, start_safari_webdriver, cleanup
import json

config_file = "game_config.json"
with open(config_file, 'r') as file:
        config = json.load(file)

# function to kill the port
def kill_port(port):
    """
    Kills the process running on the specified port.
    """
    import os
    import subprocess

    try:
        output = subprocess.check_output(["lsof", "-t", "-i", f":{port}"])
        pid = int(output.decode().strip())
        os.system(f"kill {pid}")
        print(f"Process running on port {port} terminated successfully.")
    except Exception as e:
        print(f"Failed to terminate process on port {port}:", e)

if __name__ == "__main__":
    # Start the Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process, driver = start_safari_webdriver(config_file['GAME_URL'])

    # If keyboard Interrupt occurs, cleanup the processes
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print(" Keyboard Interrupt detected. Cleaning up...")
        cleanup(server_process, safari_process)
        