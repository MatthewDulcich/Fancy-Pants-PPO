import json

def load_config(file_path):
    """
    Loads and validates the configuration file.
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    # Add any validation logic here if needed
    return config