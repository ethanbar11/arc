import sys

from omegaconf import OmegaConf
CONFIG_LOCATION = r'config/config.yaml'
conf = OmegaConf.load(CONFIG_LOCATION)
second_conf = OmegaConf.load(r'config/code_llama_cot_config.yaml')
cli_conf = OmegaConf.from_cli(sys.argv[1:])

config = OmegaConf.merge(conf,second_conf, cli_conf)
