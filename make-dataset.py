import os
from src.data.preprocessing import preprop
from pathlib import Path
import cv2
from tqdm import tqdm

def main(input_filepath, output_filepath):

    data_folders = [f for f in os.listdir(input_filepath) if Path(fr'{input_filepath}/{f}').is_dir()]

    for index, folder in enumerate(data_folders):
        if (index > 10): return
        for f in tqdm(os.listdir(fr'{input_filepath}/{folder}')):
            img = cv2.imread(fr'{input_filepath}/{folder}/{f}', cv2.IMREAD_GRAYSCALE)
            processed_image = preprop(img)
            cv2.imwrite(fr'{output_filepath}/proc_{f}', processed_image)
        
main("data/raw", "data/processed")