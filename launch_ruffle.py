import os
import signal
import subprocess
import http.server
import socketserver
import webbrowser

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

def is_port_in_use(port):
    try:
        subprocess.check_output(f"lsof -i:{port}", shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def start_server():
    # Kill any existing process on port 8000 before starting the server
    if is_port_in_use(PORT):
        kill_port(PORT)

    # Ensure the port is free before starting the server
    if not is_port_in_use(PORT):
        # Start the server
        os.chdir(DIRECTORY)
        with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
            webbrowser.open(f"http://localhost:{PORT}/launch_ruffle.html")
            print("Serving at port", PORT)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
    else:
        print(f"Port {PORT} is still in use. Please free the port and try again.")