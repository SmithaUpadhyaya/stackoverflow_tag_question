from src.entity.data_featurization import TagFeaturization, DataFeaturizationConfig, TextFeaturization
from src.train_scripts.distributed_train import define_distributed_strategy
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from src.train_scripts.model_tf import define_model, model_hyperparam
from src.config.configuration import ConfigurationManager
from sklearn.model_selection import train_test_split
from src.entity.wrd_embed_glove import WordEmbGlove
from src.utils.path_utils import create_directories
from src.utils.path_utils import get_project_root
from tensorflow_addons.metrics import F1Score
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam
from datetime import datetime
import tensorflow as tf
from src import logger
import pandas as pd
import numpy as np
import pickle
import math
import gc

if __name__ == '__main__':

    logger.info(f">>>>>> Training LSTM model started <<<<<<")

    config = ConfigurationManager()
    data_cleaning_config = config.get_data_cleaning_config()
    model_train_artifact_config = config.get_model_artifacts_config()
    word_embed_config = config.get_word_emb_config()

    cleaned_data_save_path = get_project_root().joinpath(data_cleaning_config.artifacts_dir).joinpath(data_cleaning_config.root_dir).joinpath(data_cleaning_config.cleaned_data_filename)
    data_folder = cleaned_data_save_path

    logger.info(f"[training_lstm]: Read hyper paramaters for the model.")
    hyper_param = model_hyperparam()

    #read all the hyper param and log the info
    str_hyper_par = 'Param values'
    str_hyper_par += '\n-----------------------------------------------------------------\n\n'
    str_hyper_par += f'DATA_FOLDER: {data_folder}'
    for key, value in hyper_param:
        str_hyper_par += '\n' 
        str_hyper_par += key + ": " + value         

    logger.info(f"[training_lstm]:\n{str_hyper_par}")

    if data_folder.exists():

        #Load training data
        logger.info(f'[training_lstm]: Reading data file: {data_folder} ...')
        dbset = pd.read_parquet(data_folder)
        dbset.reset_index(inplace = True)

        dbset = dbset[['clean_title', 'tags']]
        logger.info(f'[training_lstm]: Traning data shape: {dbset.shape}')

        feature_config = DataFeaturizationConfig(num_words = hyper_param['num_words'], 
                                                 max_title_len = hyper_param['max_title_len']
                                                )
        tag_feature_obj = TagFeaturization(Config = feature_config)
        text_feature_obj = TextFeaturization(Config = feature_config)
        word_embed_obj = WordEmbGlove(Config = word_embed_config)

        ###### Select the tag and generate vocabulary tokenizer, since selected tag would reduce the number of question and that will reduce vocabulary

        ### Convert tags to feature and select top n tags as feature
        logger.info(f'[training_lstm]: Converting tags into features for multilabel problems...')   

        ## Step 1: Return multilabel_binary object that is fit on the tags data 
        logger.info(f'[training_lstm]: Fit tags on MultiLabelBinarizer.')
        multilabel_binarizer_obj = tag_feature_obj.convert_tags_features(dbset.tags) 
        logger.info(f'[training_lstm]: Number of unique tags {multilabel_binarizer_obj.classes_}')

        ## Step 2: Transform tags to One-hot encoded Y label using multilabel_binary object
        logger.info(f'[training_lstm]: Transform tags using fited MultiLabelBinarizer object.')
        Y = multilabel_binarizer_obj.transform(dbset.tags.apply(lambda x: { i for i in x.split(',')}))

        logger.info(f'[training_lstm]: Selected top {str(hyper_param["tags_to_choose"])} tag features that explain most of the questions.')
        Y, selected_tags_idx = tag_feature_obj.tags_to_choose(Y, hyper_param['tags_to_choose'])
        
        selected_features_name = multilabel_binarizer_obj.classes_[selected_tags_idx[:hyper_param['tags_to_choose']]]
        logger.info(f'[training_lstm]: Selected tags- {selected_features_name} .')

        ## Step 3: Serialize multilabel_binarizer_obj to file, which can be later used for predit from model
        logger.info(f'[training_lstm]: Saving multilabel binary object started...')
        filename = "train_lstm"
        serialize_objects_dir = model_train_artifact_config.artifacts_dir.joinpath(model_train_artifact_config.serialize_objects_dir).joinpath(filename)
        create_directories([serialize_objects_dir])
        serialize_multilabel_binarizer_obj_dir = serialize_objects_dir.joinpath("multilabel_binarizer_obj.pickle")

        with open(serialize_multilabel_binarizer_obj_dir, "wb") as outfile: # "wb" argument opens the file in binary mode        
            pickle.dump(multilabel_binarizer_obj, outfile)
        logger.info(f'[training_lstm]: Saved multilable_binary object at {serialize_objects_dir}.')

        ## Step 4: Save the selected feature name list to use predict from model 
        logger.info(f'[training_lstm]: Saving selected features...')
        serialize_selected_features_names_dir = serialize_objects_dir.joinpath("selected_features_names.pickle")

        with open(serialize_selected_features_names_dir, "wb") as outfile: # "wb" argument opens the file in binary mode        
            pickle.dump(selected_features_name, outfile)

        logger.info(f'[training_lstm]: Saving selected features at {serialize_selected_features_names_dir}.')

        ## Step 5: Removing the question with no tags selected
        logger.info(f'[training_lstm]: Removing the question with no tags selected.')
        row_idx = np.nonzero(Y.sum(axis = 1))[0] #Question with nonzero tags
        Y = Y[row_idx,:]
        X = dbset.iloc[row_idx,:]
        #X = X[row_idx,:]

        logger.info(f'[training_lstm]: Number of records after selecting tag. X-{X.shape}, Y-{Y.shape}')

        ## Step 6: Int tokenizer on text
        logger.info(f'[training_lstm]: Init tokenizer for model.')
        tokenizer = text_feature_obj.init_tokenizer(X.clean_title)

        #Len fun return the actual number of words in the vocab, Then why add 1 to vocab_size? 
        # Ans: First we adding 1 to the size of the vocabulary, not to the token IDs.  
        # Tokenizer.word_index is a dictionary that contains token keys (string) and token ID values (integer), 
        # and where the first token ID is 1 (not zero) and these token IDs are assigned incrementally. 
        # Therefore, the greatest token ID in word_index is len(word_index).
        # Now when create numpy array of len(word_index) where index start with 0, but the word index start from 1.
        # So when define embed_matrix for those word_index will have 1 word index interger less. 
        # so we add 1 to len so that in embed_matrix (which contain embeding for those words mapped as interge). 
        # In embed_matrix row index 0 will be some random value which will not be used. 
        # So after mapping words to integer and then generate word vector in embed_matrix, words with max interger will also have the emebedding.
        # To handel Out-of-Vocabulary token (oov_token) if it was provided when building the tokenizer, or ignored if not. The oov token, if provided, has index 1 in word_index.
        vocab_size = len(tokenizer.word_index) + 1 

        #Get TF-IDF 1-gram for each. Weight TF_IDF word embeding 
        mapping_word2tfidf = {}
        if hyper_param['weight_tf_idf'] == True: 

            logger.info('[training_lstm]: Generate TD-IDF features for each words. Weighted word vector...')
            mapping_word2tfidf = text_feature_obj.get_tfidf_vector_feature(X.clean_title, tokenizer)

        X = text_feature_obj.get_text_features(tokenizer, X.clean_title)

        logger.info(f'[training_lstm]: Unique Word Count/Vocabulary Size: {vocab_size}') #Output value: 31767
        logger.info(f'[training_lstm]: Max length of title: {str(hyper_param["max_title_len"])} ') #Output value: 168 
        logger.info(f'[training_lstm]: Shape X: {X.shape}')

        #Unique Word Count/Vocabulary Size: 31767
        #Reduce size Unique Word Count/Vocabulary Size: 28926 after token after considering selected tags

        #Generate the embedding matrix based on the index generated
        logger.info('[training_lstm]: Generate the embedding matrix based on the index generated by tokenizer.')
        embeddings_dictionary = word_embed_obj.read_glove_word_embed(hyper_param['word_embed_size'])
        embedding_matrix = np.zeros((vocab_size, hyper_param['word_embed_size']))

        logger.info('[training_lstm]: Mapping words to token index...')
        for word, index in tokenizer.word_index.items():

            embedding_vector = embeddings_dictionary.get(word)
            
            # fetch df score
            idf = 1
            if len(mapping_word2tfidf) > 0:
                try:
                    idf = mapping_word2tfidf[str(word)]
                except:
                    idf = 0

            if embedding_vector is not None:

                embedding_vector = embedding_vector * idf #Weight vector by multiplying it with calculated tfifd for that word.
                embedding_matrix[index] = embedding_vector
        
        hyper_param['embedding_matrix'] = embedding_matrix

     
        #Train and Valid Split 
        # Initial tried shuffle = False but some data in the valid dataset always fail. So decided to shuffle the data and include those in training   
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size = hyper_param['train_split'], shuffle = True, random_state = 42)

        logger.info(f'[training_lstm]: X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
        logger.info(f'[training_lstm]: X_valid shape: {X_valid.shape}, Y_valid shape: {Y_valid.shape}')

        strategy, num_replicas_in_sync = define_distributed_strategy()

        #Define train and valid dataset
        batch_size = hyper_param['batch_size'] * num_replicas_in_sync

        train_batch = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE) 
        valid_batch = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid)).batch(batch_size).prefetch(tf.data.AUTOTUNE) 
                
        tot_train = X_train.shape[0]
        tot_valid = X_valid.shape[0]

        logger.info(f'[training_lstm]: Batch size: {batch_size}')

        #Clean unwanted variable object
        del [X_train, X_valid, Y_train, Y_valid, X, Y, dbset]
        gc.collect()

        model_logs_path = get_project_root().joinpath(model_train_artifact_config.artifacts_dir).joinpath(model_train_artifact_config.train_log_dir)

        with strategy.scope(): 

            logger.info('[training_lstm]: Define model')
            model = define_model(hyper_param)
            
            adam_optimize = Adam(learning_rate = hyper_param['lr'])

            metric_f1score = F1Score(num_classes = hyper_param['tags_to_choose'],
                                     threshold = 0.5, 
                                     average = 'micro', 
                                     name = 'f1score')
            metric_accuracy = Accuracy(name = 'acc')

            model.compile(loss = BinaryCrossentropy(from_logits = False), #from_logits = False, if output will be be sigmoid. 'binary_crossentropy',
                          optimizer = adam_optimize, 
                          metrics = [
                                    metric_f1score, 
                                    metric_accuracy
                                    ]
                        )

            ###  Callbacks
            
            #Earlystop  Callback
            earlystop_callback = EarlyStopping(monitor = 'val_f1score',
                                               mode = 'max', 
                                               verbose = 1, 
                                               patience = 100)

            #Tensorboard Callback

            ##filename += 'bidirection_lstm_layers_'
            #filename += 'lr' + str(hyper_param['lr']).replace('.','') + '_'
            #filename += 'units' + str(hyper_param['lstm_units']) + '_' 
            #filename += 'dropout(' + str(hyper_param['dropout']).replace('.','') + ',' + str(hyper_param['recurrent_dropout']).replace('.','') +')_' 
            #filename += 'l2' + str(hyper_param['l2_reg']).replace('.','') + '_'
            #filename += 'batch' + str(hyper_param['batch_size']) + '_'
            #filename += 'epoch' + str(hyper_param['epoch'])
            ##filename += datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = model_train_artifact_config.log_filename_format + "_" + datetime.now().strftime("%Y%m%d")
            if hyper_param['weight_tf_idf'] == True:
                filename += '_weight_tfidf_'
            if hyper_param['trainable_embed'] == True:
                filename += '_train_embed_'
            filename = filename.format(
                                        str(hyper_param['lr']).replace('.',''),
                                        str(hyper_param['lstm_units']), 
                                        '(' + str(hyper_param['dropout']).replace('.','') + ',' + str(hyper_param['recurrent_dropout']).replace('.','') +')',
                                        str(hyper_param['batch_size']),
                                        str(hyper_param['epoch'])
                            )
            filename += 'l2_' + str(hyper_param['l2_reg']).replace('.','')

            #Write Summary in tensorflow 
            """
            summary_writer = tf.summary.create_file_writer(log_path) 
            with summary_writer.as_default():

                #hyperparamter

                def dict2mdtable(d, key = 'Name', val = 'Value'):

                    rows = [f'| {key} | {val} |']
                    rows += ['|--|--|']
                    rows += [f'| {k} | {v} |' for k, v in d.items()]
                    return "  \n".join(rows)

                hparam_dict = hyper_param.copy()
                hparam_dict['weight_tfidf'] = WEIGHT_TF_IDF
                hparam_dict['tags_to_choose'] = TAGS_TO_CHOOSE
                hparam_dict['word_embed_size'] = WORD_EMBED_SIZE
                hparam_dict['max_title_len'] = MAX_TITLE_LEN
                summary_writer.text('Hyperparams', dict2mdtable(hparam_dict), 1)
                summary_writer.flush()
            """

            logdir = model_logs_path.joinpath(filename)
            tensorboard_callback = TensorBoard(logdir, histogram_freq = 1)

            #save model checkpoint
            model_cp_path = model_train_artifact_config.artifacts_dir.joinpath(model_train_artifact_config.checkpoint_dir).joinpath(filename)
            create_directories([model_cp_path])

            model_cp_path = get_project_root().joinpath(model_cp_path)            
            #model_cp_file = model_cp_path.joinpath('best_model.h5')
            
            modelcp_callback = ModelCheckpoint(filepath = model_cp_path, #model_cp_file,
                                               monitor = 'val_f1score', 
                                               mode = 'max', 
                                               verbose = 0, 
                                               save_freq = 'epoch')

        #print(f'LSTM Units: { str(hyper_param["lstm_units"]) } , Learning rate: { str(hyper_param["lr"]) } , Dropout: ( { str(hyper_param["dropout"]) } , { str(hyper_param["recurrent_dropout"]) } ), L2 reg: { str(hyper_param["l2_reg"]) }')

        #model.summary()


        logger.info('[training_lstm]: Training model...')

        model.fit(train_batch, validation_data = valid_batch,
                #X_train, Y_train, validation_data = (X_valid, Y_valid), 
                #batch_size = hyper_param['batch_size'] * num_replicas_in_sync, 

                steps_per_epoch = math.ceil(tot_train/batch_size),
                validation_steps = math.ceil(tot_valid/batch_size),

                epochs = hyper_param['epoch'],
                verbose = 1, #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                shuffle = True,

                callbacks = [tensorboard_callback, modelcp_callback,earlystop_callback]
          
                )

        logger.info(f">>>>>> Training LSTM model completed <<<<<<\n\nx==========x")

    else:

        logger.info(f'[training_lstm]: Training datafile does not exits: {data_folder}')

#=========================================================================


#import shutil
#shutil.rmtree('/kaggle/working/models') #remove dir with files
#shutil.rmtree('/kaggle/working/logs')

#from IPython.display import FileLink
#FileLink(r'models/checkpoints/logs_hparam_lstm_layers_lr-0_02_lstm_units-544_batch_size-2048_epoch-150/best_model.h5')

#os.listdir('/kaggle/working/models/checkpoints/logs_hparam_lstm_layers_lr-0_2_lstm_units-512_batch_size-2048_epoch-150_')





    