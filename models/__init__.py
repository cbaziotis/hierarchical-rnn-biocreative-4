#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

logging.basicConfig(
    format='%(asctime)s [%(name)-8s:%(funcName)8s] [%(levelname)-8s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__file__)


def main(args):
    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Use verbose mode.',
                        action="store_true")
    args = parser.parse_args()

    main(args)