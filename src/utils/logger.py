import logging
import logging.config
def setup_logger(config_path: str):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
