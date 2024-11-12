
#dataclass : Module provides a decorator and functions for automatically adding generated special methods __init__() and __repr__() to user-defined classes.
#            Generally when define var in class we define _init_ to init those var variable.

from src.utils.path_utils import get_project_root
from ensure import ensure_annotations
from dataclasses import dataclass
from zipfile import ZipFile
from pathlib import Path
from src import logger
import urllib.request
import numpy as np
#import os

@dataclass(frozen = True)
class WordEmbGloveConfig:

    artifacts_dir: str
    root_dir: str    
    glove_embed_url: str

class WordEmbGlove:    

    # Class variable
    config: WordEmbGloveConfig

    glove_file: str = "glove.6B.zip"
    glove_dim_50: str = "glove.6B.50d.txt"
    glove_dim_100: str = "glove.6B.100d.txt"
    glove_dim_200: str = "glove.6B.200d.txt"
    glove_dim_300: str = "glove.6B.300d.txt"

    @ensure_annotations
    def __init__(self, 
                 Config: WordEmbGloveConfig
                )-> None:
                
        self.config = Config

    @ensure_annotations
    def download_glove_embed(self, dimension: int = 0) -> Path:
        
        """
        Download glove embed zip file if not avalable
        """

        download_path = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath(self.glove_file)
        logger.info(f"[download_glove_embed]: Local glove embed zip path: [{download_path}]...")

        if not download_path.is_file():
            
            url_link = self.config.glove_embed_url + self.glove_file #'https://nlp.stanford.edu/data/glove.6B.zip'

            logger.info(f"[download_glove_embed]: Downloading the Glove embed from url: [{url_link}]...")
            local_filename, _ = urllib.request.urlretrieve(url = url_link,
                                                           filename = download_path   
                                                          )
            download_path = local_filename

        return self.extact_glove_zip(download_path, dimension)

    @ensure_annotations
    def extact_glove_zip(self, filename: Path, dim: int = 200) -> Path:

        #extract it in data folder
        #unzip "/kaggle/working/glove.6B.zip" -d "/content/" #Jupiter notebook

        #Input param
        logger.info(f"[extact_glove_zip]: Input paramaters. filename: [{filename}], dim: {dim} ...")

        #Creating a zip object
        #unzip_path = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir)
        #unzip_path = Path(filename).parent
        unzip_path = filename.parent

        logger.info(f"[extact_glove_zip]: Extacted glove embed zip path: [{unzip_path}]...")

        if dim == 0: #extact all files in the zip
                        
            if unzip_path.is_dir():

                """
                With extractall will extact following files:
                1. glove.6B.50d.txt   : txt file with represent words with 50 dimension
                2. glove.6B.100d.txt  : txt file with represent words with 100 dimension 
                3. glove.6B.200d.txt  : txt file with represent words with 200 dimension
                4. glove.6B.300d.txt  : txt file with represent words with 300 dimension
                """

                logger.info(f"[extact_glove_zip]: Extacting glove embed zip started...")

                with ZipFile(filename, 'r') as zObject:
                    zObject.extractall(path = unzip_path) #extract all the files in the folder

                logger.info(f"[extact_glove_zip]: Extacted glove embed zip completed...")
        else:        
            
            dim = self.get_filename_by_dimesion(dim)            
            
            if dim != "":

                if not unzip_path.joinpath(dim).exists():

                    logger.info(f"[extact_glove_zip]: Extacting selected dimension embed from glove zip: {dim} ...")

                    with ZipFile(filename, 'r') as zObject:                
                        zObject.extract(dim, path = unzip_path)
                        
                unzip_path = unzip_path.joinpath(dim)

                logger.info(f"[extact_glove_zip]: Glove dimension: {dim} filename: [{unzip_path}] ...")
            
            else:
                unzip_path = Path('')

        return unzip_path


    @ensure_annotations
    def get_filename_by_dimesion(self, dim: int) -> str:

        if dim == 50:
            return self.glove_dim_50
        elif dim == 100:
            return self.glove_dim_100
        elif dim == 200:
            return self.glove_dim_200
        elif dim == 300:
            return self.glove_dim_300
        else:
            return ''


    @ensure_annotations
    def get_glove_embed_file(self, dimension: int) -> Path:
        
        return self.download_glove_embed(dimension)

    @ensure_annotations
    def read_glove_word_embed(self, dimesion: int = 200) -> dict:   
        
        embeddings_dictionary = dict()

        filename = self.get_glove_embed_file(dimesion)

        logger.info(f"[read_glove_word_embed]: Reading word embeding for dimension {dimesion} from filename: [{filename}]...")

        with open(filename,'rb') as glove_file : #'/content/glove.6B.200d.txt'
            
            for line in glove_file:        
                values = line.split()
                word = values[0]
                vector_dimensions  = np.asarray(values[1:],dtype = 'float32')
                embeddings_dictionary[word] = vector_dimensions

        return embeddings_dictionary
