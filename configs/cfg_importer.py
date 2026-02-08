import yaml

def import_cfg():
    """
    Import utility for config file across project
    :return: yaml with config
    """
    try:
        with open("../configs/config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg
    except Exception as e:
        print(f"Error: Config not loaded ->{e}")