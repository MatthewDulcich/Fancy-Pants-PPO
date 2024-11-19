from game_env_setup import start_ruffle_host, start_safari_webdriver, GAME_URL, cleanup

if __name__ == "__main__":
    # Start the Ruffle host server and Safari WebDriver
    server_process = start_ruffle_host()
    safari_process, driver = start_safari_webdriver(GAME_URL)

    # If keyboard Interrupt occurs, cleanup the processes
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print(" Keyboard Interrupt detected. Cleaning up...")
        cleanup(server_process, safari_process)
        