# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

from argparse import ArgumentParser
from datetime import datetime
import config
from main_run import *
import faulthandler
import os
from config import *

faulthandler.enable()

# config logger
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.root.setLevel(logging.INFO)
logger = logging.getLogger('log')
if not os.path.exists("./logs"):
    os.makedirs("./logs")
logger.addHandler(logging.FileHandler('./logs/main_log.log', 'a'))

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '-d',
                        help='Which dataset should be used?',
                        choices={'UCR','synthetic','synthetic_hard'},
                        default='UCR')
    parser.add_argument('--complete_UCR',
                        help='Run on all UCR univariate timeseries datasets',
                        action='store_true',
                        default=False)
    parser.add_argument('--multi_scale',
                        help='Run on different scaling parameters (defined in config file)',
                        action='store_true',
                        default=False)
    parser.add_argument('--normalize',
                        help='If dataset should be normalized',
                        action='store_true',
                        default=True)
    parser.add_argument('--HDC_dim',
                        help='Defines the HDC dimension',
                        type=int,
                        default=9996)
    parser.add_argument('--scale',
                        help='scaling of the scalar encoding with fractional binding ',
                        default=0)
    parser.add_argument('--ucr_idx',
                        help='index for UCR dataset ensemble',
                        type=int,
                        default=0)
    parser.add_argument('--stat_iterations',
                        help='number of repetition for statistical evaluation',
                        default=1)
    parser.add_argument('--config',
                        help='define the config struct in config.py for more parameters',
                        default='Config_orig')
    parser.add_argument('--model',
                        help='Which model should be used?',
                        choices={'HDC_MINIROCKET','MINIROCKET'},
                        default='HDC_MINIROCKET')
    args = parser.parse_args()

    args.scale = float(args.scale)

    # init high level network trainer
    trainer = NetTrial(args)

    logger.info('_________________________' + str(datetime.now()))
    logger.info("--- " + args.model + " Model---")
    logger.info("- Dataset: " + args.dataset)
    logger.info("- Normalization: " + str(args.normalize))
    logger.info("- Config: " + str(args.config))

    if args.multi_scale:
        exec('config = ' + args.config + '()')
        logger.info("##### Run on multiple scales: " + str(config.scales))
        scales = config.scales
    else:
        scales = np.array([args.scale])

    for s_idx in range(len(scales)):
        args.scale = scales[s_idx]
        logger.info("Scale = " + str(args.scale))

        if args.complete_UCR:
            logger.info("##### Full experiment on all UCR univariate time series")
            for i in range(128):
                trainer = NetTrial(args)
                logger.info("Index: " + str(i))
                trainer.config.ucr_idx = i
                trainer.config.scale_idx = s_idx
                trainer.load_data()
                trainer.train()
                del trainer
        else:
            logger.info("#### normal Training on " + args.dataset + ": ")
            logger.info("Config: HDC_dim = " + str(args.HDC_dim) + " scale = " + str(args.scale))
            logger.info("UCR Index = " + str(args.ucr_idx))
            trainer = NetTrial(args)
            trainer.config.ucr_idx = args.ucr_idx
            trainer.config.scale_idx = s_idx
            trainer.load_data()
            trainer.train()
