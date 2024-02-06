# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
from tqdm import tqdm

import cv2
from dotenv import find_dotenv, load_dotenv
from preprocessing import preprop


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data ...')

    data_folders = [f for f in os.listdir(input_filepath) if Path(fr'{input_filepath}/{f}').is_dir()]

    for folder in data_folders:
        for f in tqdm(os.listdir(fr'{input_filepath}/{folder}')):
            img = cv2.imread(fr'{input_filepath}/{folder}/{f}', cv2.IMREAD_GRAYSCALE)
            processed_image = preprop(img)
            cv2.imwrite(fr'{output_filepath}/proc_{f}', processed_image)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
