from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID

# --- Window Management Functions ---

def get_window_list():
    """
    Retrieves a list of all on-screen windows with their titles and owner names.
    """
    window_list = []
    window_info_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    for window_info in window_info_list:
        window_title = window_info.get('kCGWindowName', 'No Title')
        window_owner_name = window_info.get('kCGWindowOwnerName', 'Unknown')
        window_layer = window_info.get('kCGWindowLayer', 0)
        window_list.append((window_title, window_owner_name, window_layer, window_info))
    return window_list


def get_most_recent_window_by_owner(owner_name):
    """
    Gets the most recent window for a given owner name.
    """
    windows = get_window_list()
    owner_windows = [window for window in windows if owner_name.lower() in window[1].lower()]
    owner_windows.sort(key=lambda x: x[2], reverse=True)  # Sort by layer
    return owner_windows[0][3] if owner_windows else None

# --- Main Script ---

if __name__ == "__main__":
    # Get the most recently opened Safari window
    safari_window = get_most_recent_window_by_owner("Safari")
    if safari_window:
        print("Safari Window Found:", safari_window)
    else:
        print("No Safari window found.")
