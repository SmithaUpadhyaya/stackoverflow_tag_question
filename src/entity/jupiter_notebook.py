from dataclasses import dataclass

@dataclass(frozen = True)
class JupiterNotebookConfig:

    artifacts_dir: str
    root_dir: str    
    data_filename: str
    cleaned_data_filename: str
