import subprocess

def mute_specific_tab(tab_url):
    """
    Mutes a specific tab in Safari based on its URL using AppleScript.
    """
    script = f"""
    tell application "Safari"
        repeat with w in windows
            repeat with t in tabs of w
                if (URL of t contains "{tab_url}") then
                    set muted of t to true
                end if
            end repeat
        end repeat
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"Tab with URL containing '{tab_url}' muted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to mute Safari tab with URL '{tab_url}':", e)