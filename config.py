import configparser
import os

my_path = os.path.dirname(__file__)
config_path = os.path.join(my_path, "config.ini")
wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "wildcards"))

def write_config():
    if os.path.exists(config_path):
        return
    
    config = configparser.ConfigParser()
    config['default'] = {
                            'wildcards_path': str(wildcards_path),
                        }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if not os.path.exists(default_conf['wildcards_path']):
            print(f"[WARN] PromptChecker wildcards_path path not found: {default_conf['wildcards_path']}. Using default path.")
            default_conf['wildcards_path'] = wildcards_path

        return {
                    'wildcards_path': default_conf['wildcards_path'] if 'wildcards_path' in default_conf else wildcards_path,
               }

    except Exception:
        print(f"[ERROR] PromptChecker wildcards_path path not found: {default_conf['wildcards_path']}. Using default path.")
        return {
            'wildcards_path': wildcards_path,
        }