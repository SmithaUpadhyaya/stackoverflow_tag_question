from src.config.configuration import ConfigurationManager
from transformers import TFBertTokenizer, TFBertModel
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from src.utils.common import def_value
from ensure import ensure_annotations
from collections import defaultdict
from pathlib import Path
import tensorflow as tf
import pickle


@ensure_annotations
def model_hyperparam() -> defaultdict:

    config = ConfigurationManager()    
    
    hyper_param = defaultdict(def_value) #For unknown key in dict it will return NONE instead of exception
    hyper_param['train_split']  = float(config.params.COMMON_PARAM.TRAIN_SPLIT)
    hyper_param['tags_to_choose'] = int(config.params.COMMON_PARAM.TAGS_TO_CHOOSE)
    hyper_param['max_title_len'] = int(config.params.COMMON_PARAM.MAX_TITLE_LEN) 
    
    if config.params.COMMON_PARAM.NUM_WORDS != 'None':
        hyper_param['num_words'] = int(config.params.COMMON_PARAM.NUM_WORDS)
    else:
        hyper_param['num_words'] = None

    hyper_param['model_name'] = str(config.params.TRANSFROMER_MODEL.BERT_MODEL_NAME)
    hyper_param['dense_units'] =  int(config.params.TRANSFROMER_MODEL.DENSE_UNITS)
    hyper_param['dropout'] = float(config.params.TRANSFROMER_MODEL.DROP_OUT)


    hyper_param['lr'] =  float(config.params.TRANSFROMER_MODEL.LR)
    hyper_param['batch_size'] =  int(config.params.TRANSFROMER_MODEL.BATCH_SIZE)
    hyper_param['epoch'] =  int(config.params.TRANSFROMER_MODEL.EPOCH)

    return hyper_param


@ensure_annotations
def model_encode(text: tf.Tensor, label: tf.Tensor, tokenizer: TFBertTokenizer, max_title_len: int):
#def model_encode(text: tf.Tensor, label: tf.Tensor, tokenizer: TFBertTokenizer, max_title_len: int, return_ylabel: bool):

    encoded = tokenizer(text,
                        max_length = max_title_len, #MAX_TITLE_LEN
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,
                        )    

    return {
            'input_ids': tf.reshape(encoded['input_ids'], [max_title_len]), #MAX_TITLE_LEN #tf.squeeze( to remove None from the shape
            'attention_mask': tf.reshape(encoded['attention_mask'], [max_title_len]) #MAX_TITLE_LEN  
        }, label

    

@ensure_annotations
def model_encode(text: tf.Tensor, tokenizer: TFBertTokenizer, max_title_len: int):

    encoded = tokenizer(text,
                        max_length = max_title_len, #MAX_TITLE_LEN
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,
                        )     
    return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        

@ensure_annotations
def model_tokenizer(hyper_param: dict) -> TFBertTokenizer:

    return TFBertTokenizer.from_pretrained(hyper_param['model_name']) 

@ensure_annotations
def model_import_tokenizer(serialize_tokenizer_obj_dir: Path) -> TFBertTokenizer:

    """
    This is not in use 
    """
    #Check if the file not found throw exception
    if serialize_tokenizer_obj_dir.exists():

       #Open and read the file
       with open(serialize_tokenizer_obj_dir,'rb') as file:

            tokenizer = pickle.load(file)
            return tokenizer
    

@ensure_annotations
def define_model(hyper_param: dict) -> Model:    

    bert_model = TFBertModel.from_pretrained(hyper_param['model_name'], name = 'tf_bert_model')

    input_ids = Input(shape = (hyper_param['max_title_len'],), name = 'input_ids', dtype = 'int32')  #MAX_TITLE_LEN
    attention_masks = Input(shape = (hyper_param['max_title_len'],), name = 'attention_mask', dtype = 'int32') #MAX_TITLE_LEN
    
    bert_embds = bert_model([input_ids, attention_masks])

    bert_output = bert_embds[1]  # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
    output = Dense(hyper_param['dense_units'], activation = 'relu', name = 'dense_units')(bert_output)
    output = Dropout(hyper_param['dropout'], name = 'dropout')(output)
    output = Dense(hyper_param['tags_to_choose'], #TAGS_TO_CHOOSE
                   activation = 'sigmoid',
                   name = 'tags_to_choose')(output)

    model = Model(inputs = [input_ids,attention_masks], outputs = output)

    return model


@ensure_annotations
def load_model_checkpoint(pre_trained_model_path: Path) -> Model:

    #This did not work. Also gave error since it consider TFBertModel as custom layer that need to be define. So tried alter approach to define the model and then load pre-trained weight

    # Model that we build uses TFBertModel as base model. We have Dense layers on top of the base model. The final model object is of type Keras.Model
    # Model saved is the completed model. We cannot use TFBertModel from_pretrained to load weights of the other Dense layer

    #model = TFBertModel.from_pretrained(pre_trained_model_path)
    model = load_model(pre_trained_model_path)

    return model