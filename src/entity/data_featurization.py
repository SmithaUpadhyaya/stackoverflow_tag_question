from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
#from src.utils.path_utils import get_project_root
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization
from ensure import ensure_annotations
from keras.utils import pad_sequences
from dataclasses import dataclass
from src import logger
import pandas as pd
import numpy as np

@dataclass(frozen = True)
class DataFeaturizationConfig:

    num_words: int = None
    max_title_len: int = 168    
    
class TagFeaturization:

    """
    Class to convert tags as features
    """
    config: DataFeaturizationConfig

    @ensure_annotations
    def __init__(self,
                 Config: DataFeaturizationConfig
                ) -> None:
        
        self.config = Config

    @ensure_annotations
    def convert_tags_features(self,
                              tag_series: pd.Series
                             ) -> MultiLabelBinarizer:

        # We can CountVectorizer with binary = True to return unique tags as feature and set value 1 if the tag exists for that datapoint.
        # Different is output type is sparse matrix. Which need to convert when passing to fit method as y label
        # vectorizer = CountVectorizer(tokenizer = lambda x: x.split(','), binary = True)
        # tag_dtm = vectorizer.fit_transform(tag_series) #dbset.tags
        # #Output feature/columns in tag_dtm : 14878


        # We can either use MultiLabelBinarizer which convert each tag to columns.
        # This required to pass tags for each datapoint as set. So applied lamdba to split tags and set the col value as set({}) and then pass it as list
        # Advantage of this is , when called transform return transformed data as numpy array which can be passed to fit method.
        # We can get value from return array of MultiLabelBinarizer object as y[:5].nonzero() # To get the values of non-zero values in the array. Retrn tuple (row, column)
        #To access individual value from the array as y[0,1631] , row=0, column: 1631

        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(list(tag_series.apply(lambda x: { i for i in x.split(',')}).values)) #dbset.tags

        return multilabel_binarizer

    @ensure_annotations
    def tags_to_choose(self,
                       multilabel_y: np.array, 
                       n: int) -> np.array:

        # Step 1: Top n sorted tags columns from the dataframe.
        t = multilabel_y.sum(axis = 0).tolist()

        # Step 2: From here we sum rows i.e count the tags we have for each question with this n tag. 
        sorted_tags_i = sorted(range(len(t)), key = lambda i: t[i], reverse = True) #retrun index of the max count. Refer: https://docs.python.org/3/howto/sorting.html
        
        # Step 3: If the tag count is zero then selected tags does not describe the question. Return count of the question with selected tag 
        multilabel_yn = multilabel_y[:,sorted_tags_i[:n]] #retun all rows with column index based on top n sorted tag index

        return multilabel_yn, sorted_tags_i

class TextFeaturization:

    """
    Class to convert text as features
    """

    config: DataFeaturizationConfig

    @ensure_annotations
    def __init__(self,
                 Config: DataFeaturizationConfig
                ) -> None:
        
        self.config = Config

    @ensure_annotations
    def init_text_vectorization_layer(self,
                                      text_series:pd.Series, 
                                      ngram:int = 1, 
                                      mode:str = 'int'
                                    ) -> TextVectorization:

        #Vectorization is the process of converting string data into a numerical representation.
        #With TextVectorization Layer we can split the text in any ngram type and also convert to any numerical represenation of choise. It's non-trainable layer. 
        #This layer must be set before training to define vocabulary, either by supplied precomputed vocab list, or by calling  adapt() method them on data.
        #Using this layer with map() in tf.data, is best for TextVectorization layer which can take the benifit of prefetch and parallel of data processing. 
        #TextVectorization layer can only be executed on a CPU, as it is mostly a dictionary lookup operation. 
        #Vocabulary contain padding token ('') and OOV token ('[UNK]')
        #Refer on all preprocessing layer in tensorflow: https://www.tensorflow.org/guide/keras/preprocessing_layers

        tokenizer_text_layer = TextVectorization(ngrams = ngram, output_mode = mode)
        tokenizer_text_layer.adapt(text_series) #"adapt" define the vocab 

        return tokenizer_text_layer

    @ensure_annotations
    def init_ngram_tokenizer(self,
                             text_series: pd.Series
                             ) -> None:
        #generate ngarm taken using nlpk. But we also need to map the ngram to integer. Which is not possible with nlpk.ngrams function
        return

    @ensure_annotations
    def init_tokenizer(self, 
                       text_series: pd.Series
                      ) -> Tokenizer:

        #Init Tokenizer object.
        #Tokenizer : it splitts a text into individual tokens. 
        #Then we need to call texts_to_sequences too convert the tokens to integer. But Tokenizer only token as ngram

        logger.info('Init Tokenizer object')
        
        #num_words:the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
        #During cleaning removed any stop words from text.
        tokenizer_obj = Tokenizer(num_words = self.config.num_words,
                                  lower = True,
                                  oov_token = '<OOV>', #Generate a token for Out-of-Vocabulary
                                )
        tokenizer_obj.fit_on_texts(text_series) #dbset.clean_title
    
        # tokenizer.index_word #Property that map int/index to word. e.g {1: 'use', 2: 'php'}
        # tokenizer.word_index #Property that map word to int/index. e.g {'use': 1, 'php': 2}
        # default unique word count/vocabulary size: 26929 

        return tokenizer_obj

    @ensure_annotations
    def get_text_features(self, 
                          tokenizer_obj: Tokenizer, 
                          text_series: pd.Series
                        ) -> pad_sequences:

        """
        transforms text data to feature_vectors that can be used in the ml model.
        tokenizer must be available.
        """
        sequences = tokenizer_obj.texts_to_sequences(text_series)
        return pad_sequences(sequences, maxlen = self.config.max_title_len)

    @ensure_annotations
    def get_tfidf_vector_feature(self,
                                 text_series: pd.Series, 
                                 tokenizer_obj: Tokenizer) -> dict:

        tfidf = TfidfVectorizer(vocabulary = list(tokenizer_obj.word_index.keys()),  #Build Tf-IDF based on the token what generated
                                smooth_idf = True,                            
                                )
        tfidf.fit(text_series)

        # dict key:word and value:tf-idf score
        # "get_feature_names_out()" return list in assending order of there idf value. 
        # i.e the feature with smallest value of idf will be first feature and largest tfidf value will be last in the list.
        # Exception if we have <OOV> or padding token then they will be as first 2 element in the list and random tfidf value is assign. 
        mapping_word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

        #mapping_word2tfidf['use'] >> 3.3886628769862273
        #mapping_word2tfidf['autopay'] >> 11.81055588854814

        return mapping_word2tfidf

    
