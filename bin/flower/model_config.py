

def get_model_config(file_path):
    """
    Reads a configuration file and returns the contents as a dictionary.
    
    Args:
        file_path (str): Path to the configuration file.
        
    Returns:
        dict: Dictionary containing the configuration parameters.
    """
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # if line strats with # or is empty, skip it
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split(',')
            # strip key and value of whitespace and remove "
            key = key.strip().replace('"', '')
            value = value.strip().replace('"', '')
            # if it is false, true or none, convert it to boolean or None
            if value.lower() == 'false':
                value = False
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'none':
                value = None
            config[key] = value
    return config