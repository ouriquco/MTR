import logging
import pprint
from mtr.datasets import WaymoDataset
from mtr.config import cfg_from_yaml_file, cfg 
from pathlib import Path

def main():
    logger = logging.getLogger("MTR_dataloader")
    CGF_PATH = "/data/cmpe258-sp24/017553289/cmpe249/MTR/tools/cfgs/waymo/mtr+20_percent_data.yaml" 

    config = cfg_from_yaml_file(CGF_PATH, cfg)
    config.ROOT_DIR = Path('/data/cmpe258-sp24/017553289/cmpe249/MTR/')
    #pprint.pp(config)

    dataset = WaymoDataset(config.DATA_CONFIG, training=False, logger=logger) 
    DATA_SAMPLE = dataset.collate_batch([dataset[0]])['input_dict']
    pprint.pp(DATA_SAMPLE)

if __name__=='__main__': 
    main()
