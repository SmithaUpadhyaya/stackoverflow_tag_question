from src.entity.jupiter_notebook import JupiterNotebookConfig
from src.entity.data_validation import DataValidationConfig
from src.entity.wrd_embed_glove import WordEmbGloveConfig #WordEmbGlove
from src.entity.data_ingestion import DataIngestionConfig
from src.entity.model_training import ModelTrainingConfig
from src.entity.data_cleaning import DataCleaningConfig
from src.utils.path_utils import create_directories
from src.utils.common import read_yaml
from ensure import ensure_annotations
from src.constants import *
import os

class ConfigurationManager:

    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH,
                 notebook_artifacts = False,
                ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        self.notebook_artifacts = notebook_artifacts #While working in notebooks all data artifacts will be in notebook

        create_directories([self.config.artifacts_root_dir])

    @ensure_annotations
    def get_word_emb_config(self) -> WordEmbGloveConfig:
        
        working_dir = os.path.join(self.config.artifacts_root_dir, self.config.word_embeding.root_dir)
        create_directories([ working_dir ])

        word_embed_config = WordEmbGloveConfig(artifacts_dir = self.config.artifacts_root_dir,
                                               root_dir = self.config.word_embeding.root_dir,
                                               glove_embed_url = self.config.word_embeding.glove_embed_url,
        )
        #word_embed_obj = WordEmbGlove(Config = word_embed_config)
            
        return word_embed_config

    @ensure_annotations
    def get_notebook_data_config(self) -> JupiterNotebookConfig:

        working_dir = os.path.join(self.config.artifacts_root_dir, self.config.notebook_artifacts.root_dir)

        create_directories([ working_dir ])

        jupiter_notebook_config = JupiterNotebookConfig(artifacts_dir = self.config.artifacts_root_dir,
                                                        root_dir = self.config.notebook_artifacts.root_dir,
                                                        data_filename = self.config.notebook_artifacts.data_filename,
                                                        cleaned_data_filename = self.config.notebook_artifacts.cleaned_data_filename,
                                                    )

        return jupiter_notebook_config

    @ensure_annotations
    def get_data_ingestion_config(self) -> DataIngestionConfig:

        if self.notebook_artifacts == True:
            working_dir = os.path.join(self.config.artifacts_root_dir, self.config.notebook_artifacts.root_dir)
            root_dir = self.config.notebook_artifacts.root_dir
        else:
            working_dir = os.path.join(self.config.artifacts_root_dir, self.config.data_ingestion.root_dir)
            root_dir = self.config.data_ingestion.root_dir

        create_directories([ working_dir ])

        data_ingestion_config = DataIngestionConfig(artifacts_dir = self.config.artifacts_root_dir,
                                                    root_dir = root_dir,  
                                                    data_downlad_url = self.config.data_ingestion.data_downlad_url,
                                                    zip_data = self.config.data_ingestion.zip_data,
                                                    data_filename = self.config.data_ingestion.data_filename,
                                                    data_test_filename = self.config.data_ingestion.data_test_filename

        )

        return data_ingestion_config

    @ensure_annotations
    def get_data_cleaning_config(self) -> DataCleaningConfig:

        if self.notebook_artifacts == True:
            working_dir = os.path.join(self.config.artifacts_root_dir, self.config.notebook_artifacts.root_dir)
            root_dir = self.config.notebook_artifacts.root_dir
        else:
            working_dir = os.path.join(self.config.artifacts_root_dir, self.config.data_cleaning.root_dir)
            root_dir = self.config.data_cleaning.root_dir

        create_directories([ working_dir ])

        tag_col = list(self.schema.COLUMNS.tags.keys()) #tuple
        selected_col = list(self.schema.COLUMNS.features.keys()) + tag_col #tuple
        
        data_cleaning_config = DataCleaningConfig(artifacts_dir = self.config.artifacts_root_dir,
                                                  root_dir = root_dir,
                                                  selected_col = selected_col,
                                                  tag_col = tag_col,
                                                  cleaned_data_filename = self.config.data_cleaning.cleaned_data_filename

        )

        return data_cleaning_config

    @ensure_annotations
    def get_data_validation_config(self) -> DataValidationConfig:

        tag_col = list(self.schema.COLUMNS.tags.keys())
        selected_col = list(self.schema.COLUMNS.features.keys())
        all_schema = self.schema.COLUMNS
   
        if self.notebook_artifacts == True:
            root_dir = self.config.notebook_artifacts.root_dir
        else:
            root_dir = self.config.data_validation.root_dir

        data_validation_config = DataValidationConfig(
                                                     col_features = selected_col,
                                                     col_tags = tag_col,
                                                     all_schema = all_schema,                            
                                                     artifacts_dir = self.config.artifacts_root_dir,
                                                     root_dir = root_dir,
                                                     validation_report = self.config.data_validation.validation_report,
        )

        return data_validation_config

    @ensure_annotations
    def get_model_artifacts_config(self) -> ModelTrainingConfig:
        
        working_dir = os.path.join(self.config.artifacts_root_dir, self.config.model_training.root_dir)
        checkpoint_dir = os.path.join(self.config.artifacts_root_dir, self.config.model_training.root_dir, self.config.model_training.checkpoint_dir)
        training_logs_dir = os.path.join(self.config.artifacts_root_dir, self.config.model_training.root_dir, self.config.model_training.train_log_dir)
        serialize_objects_dir = os.path.join(self.config.artifacts_root_dir, self.config.model_training.root_dir, self.config.model_training.serialize_objects_dir)

        create_directories([ working_dir, checkpoint_dir, training_logs_dir, serialize_objects_dir ])

        model_training_config = ModelTrainingConfig(
                                                    artifacts_dir = self.config.artifacts_root_dir,
                                                    root_dir = self.config.model_training.root_dir,
                                                    checkpoint_dir = os.path.join(self.config.model_training.root_dir, self.config.model_training.checkpoint_dir),
                                                    train_log_dir = os.path.join(self.config.model_training.root_dir, self.config.model_training.train_log_dir),
                                                    serialize_objects_dir = os.path.join(self.config.model_training.root_dir, self.config.model_training.serialize_objects_dir),

                                                    log_filename_format = self.config.model_training.log_filename_format,
        )
            
        return model_training_config