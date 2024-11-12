from src.utils.path_utils import get_project_root
from ensure import ensure_annotations
from dataclasses import dataclass
from zipfile import ZipFile
from pathlib import Path
from src import logger
import pandas as pd


@dataclass(frozen = True)
class DataIngestionConfig:

    artifacts_dir: str
    root_dir: str    
    data_downlad_url: str
    zip_data: str
    data_filename: str
    data_test_filename: str


class DataIngestion:

    config: DataIngestionConfig

    @ensure_annotations
    def __init__(self, 
                 Config: DataIngestionConfig
                )-> None:
                
        self.config = Config

    def unzip_db(self):

        try:

            zip_filename = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath(self.config.zip_data)

            logger.info(f"[unzip_db]: Unzip datafile: [{zip_filename}]...")

            if zip_filename.exists():

                logger.info(f"[unzip_db]: Unzip data file does not exists...")

                unzip_path = zip_filename.parent
                logger.info(f"[unzip_db]: Unzip data file at [{unzip_path}]...")
                
                with ZipFile(zip_filename, 'r') as zObject:
                    zObject.extractall(path = unzip_path)

                logger.info(f"[unzip_db]: Extacted sucessfully...")

            else:
                logger.info(f"[unzip_db]: Unzip datafile: [{zip_filename}] not found...")

        except Exception as e:
            raise e

    @ensure_annotations
    def combine_train_test_dbset(self) -> Path:

        train_file = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath('train.csv')

        test_file = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath('test.csv')

        merge_file = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath(self.config.data_filename)
            
        if train_file.exists() and test_file.exists():

            logger.info(f"[combine_train_test_dbset]: Merge train and test files...")
            df_train = pd.read_csv(train_file, encoding = 'latin1') #Total 63462 records in train
            #df_train.info()

            df_test = pd.read_csv(test_file, encoding = 'latin1') #Total 39678 records in train
            #df_test.info()

            #Let's merge data from train and test to have a larger dataset.
            df_merge = pd.concat([df_train, df_test], axis = 0) #Total 103140 records in train
            #df_merge.info()

            df_merge.sort_values(by = 'PostCreationDate' , inplace = True)
            #df_merge.head(3)
            #df_merge.tail(3)

            df_merge.to_parquet(merge_file, engine = 'fastparquet')

        else:
            
            logger.info(f"[combine_train_test_dbset]: Multiple file does not exists...")

            if train_file.exists(): #rename the train file name

                logger.info(f"[combine_train_test_dbset]: Rename train.csv file to {self.config.data_filename}...")
                train_file.rename(self.config.data_filename)
            
            elif test_file.exists(): #rename the test file name 

                logger.info(f"[combine_train_test_dbset]: Rename test.csv file to {self.config.data_filename}...")
                test_file.rename(self.config.data_filename)

        logger.info(f"[combine_train_test_dbset]: Combine filename at [{merge_file}]...")

        return merge_file
