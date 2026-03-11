import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            
            if not os.path.exists(zip_download_dir):
                logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
                file_id = dataset_url.split("/")[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix+file_id,zip_download_dir)
                logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(zip_download_dir))}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # Check if the extracted data directory already exists and is not empty
        extracted_data_path = os.path.join(unzip_path, "kidney-ct-scan-image")
        if not os.path.exists(extracted_data_path) or not os.listdir(extracted_data_path):
            logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        else:
            logger.info(f"Data already extracted at {extracted_data_path}")