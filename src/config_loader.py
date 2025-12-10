import yaml
import os

_CONFIG = None

def load_config():
    global _CONFIG
    if _CONFIG is None:
        config_path = os.environ.get('CONFIG_PATH','config.yaml')

        with open(config_path, 'r') as f:
            _CONFIG = yaml.safe_load(f)
        
    return _CONFIG