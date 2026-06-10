import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

class ConfigurationManager:

    def __init__(self,
                 config_filepath = ROOT_DIR / "configs" / "config.yaml"):
        
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

    def get_config(self):
        return self.config