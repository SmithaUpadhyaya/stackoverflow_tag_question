from src.entity.data_ingestion import DataIngestionConfig
from src.utils.path_utils import get_project_root
from ensure import ensure_annotations
from dataclasses import dataclass
from src import logger
import pandas as pd
import gc

@dataclass(frozen = True)
class DataValidationConfig:

    col_features: list
    col_tags: list    
    all_schema: dict

    artifacts_dir: str
    root_dir: str    
    validation_report: str
    

class DataValidation:

    config: DataValidationConfig
    data_config: DataIngestionConfig

    #@ensure_annotations #Commented it since giving error "isinstance() arg 2 must be a type, tuple of types" for list object in the Config class 
    def __init__(self, 
                 Config: DataValidationConfig, 
                 Data_Config: DataIngestionConfig
                ) -> None:        

        self.config = Config
        self.data_config = Data_Config

    @ensure_annotations
    def validate_all_columns(self)-> bool:

        try:
            
            logger.info(f"[validate_all_columns]: Validation of features started...")

            validation_status = True
            validation_status_report_file = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath(self.config.validation_report)
             
            filename = get_project_root().joinpath(self.data_config.artifacts_dir).joinpath(self.data_config.root_dir).joinpath(self.data_config.data_filename)
            dbset = pd.read_parquet(filename, engine = 'fastparquet')

            all_cols = list(dbset.columns)
            all_schema = self.config.col_features + self.config.col_tags
            
            #Check if the required feature columns exits in the dataset
            for col in all_schema:
                if col not in all_cols:

                    validation_status = False
                    with open(validation_status_report_file, 'w') as f:
                        f.write(f'Validation status: {validation_status}, Missing required feature columns: "{col}"')
                
            
            #Check the dtype of the required features
            dbset_dtypes = dbset.dtypes

            for col in self.config.col_features:
                if self.config.all_schema.features[col] != str(dbset_dtypes[col]):

                    validation_status = False
                    with open(validation_status_report_file, 'w') as f:
                        f.write(f'Validation status: {validation_status}, Schema for feature columns: "{col}" does not match the dataset. Config schema required "{self.config.all_schema.features[col]}" found "{str(dbset_dtypes[col])}" ')


            del [dbset]
            gc.collect()

            if validation_status == False:
                logger.info(f"[validate_all_columns]: Validation failed. Refer [{validation_status_report_file}] for the list of issues...")
            else:
                logger.info(f"[validate_all_columns]: Validation sucessfully.")

            logger.info(f"[validate_all_columns]: Validation of features completed...")

            return validation_status
        
        except Exception as e:
            raise e