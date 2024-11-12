
from keras.layers import Embedding, Bidirectional, LSTM, Dense,GlobalAveragePooling1D, Dropout
from src.config.configuration import ConfigurationManager
from keras.initializers import Constant
from src.utils.common import def_value
from ensure import ensure_annotations
from collections import defaultdict
from keras.models import Sequential
from keras.regularizers import L2
from keras.models import Model

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

    hyper_param['lstm_units'] =  int(config.params.LSTM_MODEL.LSTM_UNITS)
    hyper_param['dropout'] = float(config.params.LSTM_MODEL.DROP_OUT)
    hyper_param['recurrent_dropout'] = float(config.params.LSTM_MODEL.RECURRENT_DROP_OUT)
    hyper_param['l2_reg'] = float(config.params.LSTM_MODEL.L2_REG)
    hyper_param['lr'] =  float(config.params.LSTM_MODEL.LR)
    hyper_param['trainable_embed'] =  bool(config.params.LSTM_MODEL.TRAINABLE_EMBED)
    hyper_param['batch_size'] =  int(config.params.LSTM_MODEL.BATCH_SIZE)
    hyper_param['epoch'] =  int(config.params.LSTM_MODEL.EPOCH)
    hyper_param['word_embed_size'] = int(config.params.LSTM_MODEL.MAX_WORD_EMBED_SIZE)
    hyper_param['weight_tf_idf'] = bool(config.params.LSTM_MODEL.WEIGHT_TF_IDF_APPLY)
    
    return hyper_param

@ensure_annotations
def define_model(hyper_param: dict) -> Model: 

    embedding_matrix = hyper_param['embedding_matrix']
    vocab_size = embedding_matrix.shape[0]
    word_embed_size = embedding_matrix.shape[1]

    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, 
                        input_length = hyper_param['max_title_len'], #MAX_TITLE_LEN
                        output_dim = word_embed_size, 
                        embeddings_initializer = Constant(embedding_matrix), 
                        #weights = [embedding_matrix],
                        trainable = hyper_param['trainable_embed'])
            ) 
    # model.add(Conv1D(2048, 
    #                  kernel_size = 3, 
    #                  padding = "valid", 
    #                  kernel_initializer = "glorot_uniform")
    #          )
    model.add(Bidirectional(
              LSTM(units = hyper_param['lstm_units'], 
                   return_sequences = True,
                   use_bias = True,
                   dropout = hyper_param['dropout'], 
                   recurrent_dropout = hyper_param['recurrent_dropout'],
                   kernel_regularizer = L2(hyper_param['l2_reg']),
                   bias_regularizer = L2(hyper_param['l2_reg']),
            )))  
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(hyper_param['dropout']))
    model.add(Dense(hyper_param['tags_to_choose'],  #TAGS_TO_CHOOSE
                    activation = 'sigmoid')
            )

    return model