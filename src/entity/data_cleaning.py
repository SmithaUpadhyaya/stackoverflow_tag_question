from src.entity.data_ingestion import DataIngestionConfig
from src.utils.path_utils import get_project_root
from nltk.tokenize import word_tokenize
from ensure import ensure_annotations
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from dataclasses import dataclass
from pathlib import Path
from src import logger
import pandas as pd
import nltk
import re

@dataclass(frozen = True)
class DataCleaningConfig:

    artifacts_dir: str
    root_dir: str    
    cleaned_data_filename: str
    selected_col: list #tuple
    tag_col: list #tuple


class DataCleaning:

    config: DataCleaningConfig
    data_config: DataIngestionConfig

    #@ensure_annotations #Commented it since giving error "isinstance() arg 2 must be a type, tuple of types" for list object in the Config class  
    def __init__(self, 
                 Config: DataCleaningConfig, 
                 Data_Config: DataIngestionConfig
                )-> None:

        self.config = Config
        self.data_config = Data_Config

    @ensure_annotations
    def load_data(self) -> pd.DataFrame:

        try:

            filename = get_project_root().joinpath(self.data_config.artifacts_dir).joinpath(self.data_config.root_dir).joinpath(self.data_config.data_filename)

            logger.info(f"[load_data]: Loading data: [{filename}] started ...")

            dbset = pd.read_parquet(filename, engine = 'fastparquet')

            logger.info(f"[load_data]: Loading data completed ...")

            return dbset

        except Exception as e:

            logger.error(f"[load_data]: Error: {e}")
            raise e

    @ensure_annotations
    def cleaning(self) -> Path:

        dbset = self.load_data()
        
        #1: Work with selected columns

        logger.info(f"[cleaning]: Checking if the selected columns exists...")
        #check if the selected columns exists
        selected_col = []
        dbset_cols = dbset.columns
        for col in self.config.selected_col:  #list()
            if col in dbset_cols:
                selected_col.append(col)
            elif col.lower() in all_lower(dbset_cols):
                selected_col.append(col.lower())

        logger.info(f"[cleaning]: Checking if the selected tags columns exists...")
        tag_col = []
        for col in self.config.tag_col:  #list()
            if col in dbset:
                tag_col.append(col)


        logger.info(f"[cleaning]: Work with selected columns {selected_col}...") 
        dbset = dbset[selected_col]

        #2: Drop records where all value are null
        logger.info(f"[cleaning]: Droping records where all value are null...")
        dbset.dropna(axis = 0, how = 'all', inplace = True)

        #3: Drop records where all values are null in the Tags columns
        logger.info(f"[cleaning]: Droping records where all values are null in the Tags columns...")        
        dbset.dropna(axis = 0, subset = tag_col , inplace = True, how = 'all')

        #4: Drop records where title and body is blank
        logger.info(f"[cleaning]: Droping records where title and body is blank...")
        ##### 1: replace blank as NaN
        dbset.replace(r'^s*$', float('NaN'), regex = True)

        ##### 2: drop records with NaN in Title and BodyMarkdown
        dbset.dropna(axis = 0, subset = ['Title', 'BodyMarkdown'] , inplace = True, how = 'all')

        #5: Fill tags with nan values as blank
        logger.info(f"[cleaning]: Fill tags with nan values as blank...")
        dbset.fillna("", inplace = True)
        #dbset.info()

        #6: Merge tags feature into single feature
        logger.info(f"[cleaning]: Combine multiple tag features in to single feature...")
        dbset.reset_index(inplace = True, drop = True)
        dbset[['tags']]  = pd.DataFrame(list(dbset[tag_col].apply(self.combine_tags, axis = 1))) #self.config.tag_col

        #7: Drop unwanted and Rename features   
        logger.info(f"[cleaning]: Drop multiple tag feature and renaming the required feature...")      
        dbset.drop(columns = tag_col, inplace = True) #self.config.tag_col
        dbset.rename(columns = {'Title': 'title', 'BodyMarkdown': 'body'}, inplace = True)

        #8: Drop duplicates
        #Final features: ['title', 'body', 'tags']
        logger.info(f"[cleaning]: Checking for duplicate records in the final features...")
        dbset.drop_duplicates(subset = ['title', 'body', 'tags'], keep = 'first', inplace = True)
        dbset.reset_index(inplace = True, drop = True)


        #9: Clean text data       
        logger.info(f'[cleaning]: Clean text feature "title" started...')

        logger.info(f'[cleaning]: Downloading stopwords from nltk...')
        #Init stop_words. Check if 'C' is a part of stopword. It was not.
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        #Init Stemmer. We can use other library for stemmer.
        logger.info(f'[cleaning]: Downloading and initilize stemmer library from nltk...')
        nltk.download("punkt")
        ps = PorterStemmer()

        def striphtml(data):

            # Since all html tag start with '<' and end with '>'. 
            # So using regexpression we find any thing in the text that maths the pattern and replace with blank.
            logger.info(f'[striphtml]: Remove html tags from the text...')
            cleanr = re.compile('<.*?>') 
            cleantext = re.sub(cleanr, ' ', str(data))
            return cleantext

        def decontracted(phrase):

            logger.info(f'[decontracted]: Decontracted words...')

            # specific
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)

            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)

            return phrase

        def clean_title(text):

            #remove html
            text = striphtml(text)   

            #convert to_lower
            text = text.lower()

            #decontracted of text
            text = decontracted(text)

            #remove special char
            #special_char = string.punctuation
            #special_char = special_char.replace('#','')
            logger.info(f'[clean_title]: Remove special chars...')
            special_char = '[!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"]'
            text = re.sub(special_char, '', text)

            #remove numberic
            logger.info(f'[clean_title]: Remove numeric from text...')
            text = re.sub("[^a-zA-Z]", " ", text)

            #removing all single letter and stopwords from question
            logger.info(f'[clean_title]: Remove all single letter and stopwords from text...')
            words = word_tokenize(text)
            text = ' '.join(str(ps.stem(j)) for j in words if j not in stop_words and len(j)!=1)    
            
            return text

        def all_lower(my_list):
            return [x.lower() for x in my_list]

        #Clean title text        
        dbset['clean_title'] = dbset.title.apply(clean_title)

        #Rename and Drop column
        logger.info(f'[cleaning]: Cleaning text data started...')
        dbset.drop(columns = ['title','body'], inplace = True)
        dbset.rename(columns = {'clean_title': 'title'}, inplace = True)

        #Save to file        
        cleaned_data_save_path = get_project_root().joinpath(self.config.artifacts_dir).joinpath(self.config.root_dir).joinpath(self.config.cleaned_data_filename)
        logger.info(f'[cleaning]: Saving cleaning data at [{cleaned_data_save_path}] started...')
        dbset.to_parquet(cleaned_data_save_path, engine = 'fastparquet')

        logger.info(f'[cleaning]: Cleaning data completed...')

        return cleaned_data_save_path

    @ensure_annotations
    def combine_tags(self, dbrow) -> list:

        """
        Combine multiple tags columns into 1 column comma seperated 
        """
        
        tags = []
        
        for i in range(0,5):

            #tag = dbrow['Tag'+ str(i+1)]
            tag = dbrow[i].strip()
            
            if len(tag) > 0:

                #Remove tag which are duplicate
                if tag not in tags:
                    tags.append(tag)        
        
        return [','.join(tags)] #return ','.join(tags)
    