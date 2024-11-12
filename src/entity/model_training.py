from dataclasses import dataclass

@dataclass(frozen = True)
class ModelTrainingConfig:

    artifacts_dir: str
    root_dir: str    
    checkpoint_dir: str
    train_log_dir: str
    serialize_objects_dir: str
    log_filename_format: str