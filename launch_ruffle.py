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

# Kill any existing process on port 8000 before starting the server
kill_port(PORT)

# Start the server
os.chdir(DIRECTORY)
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    webbrowser.open(f"http://localhost:{PORT}/launch_ruffle.html")
    print("Serving at port", PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")