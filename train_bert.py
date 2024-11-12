from src.train_scripts.model_bert import define_model, model_encode, model_tokenizer, model_hyperparam
from src.entity.data_featurization import TagFeaturization, DataFeaturizationConfig
from src.train_scripts.distributed_train import define_distributed_strategy
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from src.config.configuration import ConfigurationManager
from sklearn.model_selection import train_test_split
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

    logger.info(f">>>>>> Training BERT model started <<<<<<")

    config = ConfigurationManager()
    data_cleaning_config = config.get_data_cleaning_config()
    model_train_artifact_config = config.get_model_artifacts_config()

    cleaned_data_save_path = get_project_root().joinpath(data_cleaning_config.artifacts_dir).joinpath(data_cleaning_config.root_dir).joinpath(data_cleaning_config.cleaned_data_filename)
    data_folder = cleaned_data_save_path

    logger.info(f"[training_bert]: Read hyper paramaters for the model.")
    hyper_param = model_hyperparam()
    
    #read all the hyper param and log the info
    str_hyper_par = 'Param values'
    str_hyper_par += '\n-----------------------------------------------------------------\n\n'
    str_hyper_par += f'DATA_FOLDER: {data_folder}'
    for key, value in hyper_param:
        str_hyper_par += '\n' 
        str_hyper_par += key + ": " + value         

    logger.info(f"[training_bert]:\n{str_hyper_par}")

    if data_folder.exists():

        #Load training data
        logger.info(f'[training_bert]: Reading data file: {data_folder} ...')
        dbset = pd.read_parquet(data_folder)
        dbset.reset_index(inplace = True)

        dbset = dbset[['clean_title', 'tags']]
        logger.info(f'[training_bert]: Traning data shape: {dbset.shape}')

        feature_config = DataFeaturizationConfig(num_words = hyper_param['num_words'], 
                                                 max_title_len = hyper_param['max_title_len']
                                                )
        tag_feature_obj = TagFeaturization(Config = feature_config)


        ###### Select the tag and generate vocabulary tokenizer, since selected tag would reduce the number of question and that will reduce vocabulary

        ### Convert tags to feature and select top n tags as feature
        logger.info(f'[training_bert]: Converting tags into features for multilabel problems...')     

        ## Step 1: Return multilabel_binary object that is fit on the tags data 
        logger.info(f'[training_bert]: Fit tags on MultiLabelBinarizer.')
        multilabel_binarizer_obj = tag_feature_obj.convert_tags_features(dbset.tags) 
        logger.info(f'[training_bert]: Number of unique tags {multilabel_binarizer_obj.classes_}')

        ## Step 2: Transform tags to One-hot encoded Y label using multilabel_binary object
        logger.info(f'[training_bert]: Transform tags using fited MultiLabelBinarizer object.')
        Y = multilabel_binarizer_obj.transform(dbset.tags.apply(lambda x: { i for i in x.split(',')}))

        logger.info(f'[training_bert]: Selected top {str(hyper_param["tags_to_choose"])} tag features that explain most of the questions.')
        Y, selected_tags_idx = tag_feature_obj.tags_to_choose(Y, hyper_param['tags_to_choose'])

        selected_features_name = multilabel_binarizer_obj.classes_[selected_tags_idx[:hyper_param['tags_to_choose']]]
        logger.info(f'[training_bert]: Selected tags- {selected_features_name} .')

        ## Step 3: Serialize multilabel_binarizer_obj to file, which can be later used for predict from model
        logger.info(f'[training_bert]: Saving multilabel binary object started...')
        foldername = "train_bert"
        serialize_objects_dir = get_project_root().joinpath(model_train_artifact_config.artifacts_dir).joinpath(model_train_artifact_config.serialize_objects_dir).joinpath(foldername)
        create_directories([serialize_objects_dir])
        serialize_multilabel_binarizer_obj_dir = serialize_objects_dir.joinpath("multilabel_binarizer_obj.pickle")

        with open(serialize_multilabel_binarizer_obj_dir, "wb") as outfile: # "wb" argument opens the file in binary mode        
            pickle.dump(multilabel_binarizer_obj, outfile)
        logger.info(f'[training_bert]: Saved multilable_binary object at {serialize_objects_dir}.')

        ## Step 4: Save the selected feature name list to use predict from model 
        logger.info(f'[training_bert]: Saving selected features...')
        serialize_selected_features_names_dir = serialize_objects_dir.joinpath("selected_features_names.pickle")

        with open(serialize_selected_features_names_dir, "wb") as outfile: # "wb" argument opens the file in binary mode        
            pickle.dump(selected_features_name, outfile)

        logger.info(f'[training_bert]: Saving selected features at {serialize_selected_features_names_dir}.')
        

        ## Step 5: Removing the question with no tags selected
        logger.info(f'[training_bert]: Removing the question with no tags selected.')
        row_idx = np.nonzero(Y.sum(axis = 1))[0] #Question with nonzero tags
        Y = Y[row_idx,:]
        X = dbset.iloc[row_idx,:]
        #X = X[row_idx,:]

        logger.info(f'[training_bert]: Number of records after selecting tag. X-{X.shape}, Y-{Y.shape}.')

        #logger.info('Map label to index and index to label.')
        #id2label = {idx:label for idx, label in enumerate(selected_features_name)}
        #label2id = {label:idx for idx, label in enumerate(selected_features_name)}

        ## Step 6: Int tokenizer on text        
        logger.info(f'[training_bert]: Init tokenizer for model {hyper_param["model_name"]}.')
        tokenizer = model_tokenizer(hyper_param)  
     
        #Not require as we can always load Tokenizer of the MODEL from Transformer
        #logger.info(f'[training_bert]: Saving tokenizer object started...')
        #serialize_tokenizer_obj_dir = serialize_objects_dir.joinpath("tokenizer_obj.pickle")
        #with open(serialize_tokenizer_obj_dir, "wb") as outfile: # "wb" argument opens the file in binary mode        
        #    pickle.dump(tokenizer, outfile)
        #logger.info(f'[training_bert]: Saved tokenizer object at {serialize_tokenizer_obj_dir}.')

        #Train and Valid Split 
        # Initial tried shuffle = False but some data in the valid dataset always fail. So decided to shuffle the data and include those in training   
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size = hyper_param['train_split'], shuffle = True, random_state = 42)

        logger.info(f'[training_bert]: X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
        logger.info(f'[training_bert]: X_valid shape: {X_valid.shape}, Y_valid shape: {Y_valid.shape}')

        strategy, num_replicas_in_sync = define_distributed_strategy()

        #Define train and valid dataset
        batch_size = hyper_param['batch_size'] * num_replicas_in_sync

        train_batch = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).map(lambda x, y: model_encode(x, y, tokenizer, hyper_param['max_title_len']), num_parallel_calls = tf.data.experimental.AUTOTUNE).prefetch(tf.data.AUTOTUNE).batch(batch_size) 
        valid_batch = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid)).map(lambda x, y: model_encode(x, y, tokenizer, hyper_param['max_title_len']), num_parallel_calls = tf.data.experimental.AUTOTUNE).prefetch(tf.data.AUTOTUNE).batch(batch_size)
                
        tot_train = X_train.shape[0]
        tot_valid = X_valid.shape[0]

        logger.info(f'[training_bert]: Batch size: {batch_size}')
        
        #Clean unwanted variable object
        del [X_train, X_valid, Y_train, Y_valid, X, Y, dbset]
        gc.collect()

        

        with strategy.scope(): 

            logger.info('[training_bert]: Define model')
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
            
            #Earlystop  Callback
            earlystop_callback = EarlyStopping(monitor = 'val_f1score',
                                               mode = 'max', 
                                               verbose = 1, 
                                               patience = 100)

            #Tensorboard Callback

            model_logs_path = get_project_root().joinpath(model_train_artifact_config.artifacts_dir).joinpath(model_train_artifact_config.train_log_dir)

            #foldername = 'logs_hparam_'
            #foldername += 'lr' + str(hyper_param['lr']).replace('.','') + '_'
            #foldername += 'units' + str(hyper_param['dense_units']) + '_' 
            #foldername += 'dropout' + str(hyper_param['dropout']).replace('.','').replace('.','') + '_' 
            #foldername += 'batch' + str(hyper_param['batch_size']) + '_'
            #foldername += 'epoch' + str(hyper_param['epoch'])
            ##foldername += datetime.now().strftime("%Y%m%d-%H%M%S")            
            foldername = model_train_artifact_config.log_filename_format + "_" + datetime.now().strftime("%Y%m%d")
            foldername = foldername.format(
                                str(hyper_param['lr']).replace('.',''), 
                                str(hyper_param['dense_units']), 
                                str(hyper_param['dropout']).replace('.','').replace('.',''),
                                str(hyper_param['batch_size']),
                                str(hyper_param['epoch'])
                            )
                        
            #log_filename_format.format(str("0.1".replace('.','')), str(123), str("0.3".replace('.','').replace('.','')), str(123), str(100))
            #Write Summary in tensorflow 
            """
            summary_writer = tf.summary.create_file_writer(model_logs_path) 
            with summary_writer.as_default():

                #hyperparamter

                def dict2mdtable(d, key = 'Name', val = 'Value'):

                    rows = [f'| {key} | {val} |']
                    rows += ['|--|--|']
                    rows += [f'| {k} | {v} |' for k, v in d.items()]
                    return "  \n".join(rows)

                hparam_dict = hyper_param
                hparam_dict['weight_tfidf'] = WEIGHT_TF_IDF
                hparam_dict['tags_to_choose'] = TAGS_TO_CHOOSE
                hparam_dict['word_embed_size'] = WORD_EMBED_SIZEn
                hparam_dict['max_title_len'] = MAX_TITLE_LEN
                summary_writer.text('Hyperparams', dict2mdtable(hparam_dict), 1)
                summary_writer.flush()
            """
            
            logdir = model_logs_path.joinpath(foldername)
            tensorboard_callback = TensorBoard(logdir, histogram_freq = 1)

            #Save Model Checkpoint
            model_cp_path = model_train_artifact_config.artifacts_dir.joinpath(model_train_artifact_config.checkpoint_dir).joinpath(foldername)
            create_directories([model_cp_path])

            model_cp_path = get_project_root().joinpath(model_cp_path)  

            #model_cp_file = model_cp_path.joinpath('best_model.h5')
            model_cp_file = model_cp_path.joinpath(model_cp_path, 'best_model.weights.h5')

            modelcp_callback = ModelCheckpoint(filepath = model_cp_file,
                                               monitor = 'val_f1score', 
                                               mode = 'max', 
                                               verbose = 1, #0
                                               save_freq = 'epoch',
                                               save_best_only = True,
                                               save_weights_only = True,
                                               )

        #model.summary()


        logger.info('[training_bert]: Training model...')

        model.fit(train_batch, validation_data = valid_batch,
                #X_train, Y_train, validation_data = (X_valid, Y_valid), 
                #batch_size = hyper_param['batch_size'] * num_replicas_in_sync, 

                steps_per_epoch = math.ceil(tot_train/batch_size),
                validation_steps = math.ceil(tot_valid/batch_size),

                epochs = hyper_param['epoch'],
                verbose = 1, #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                shuffle = True,

                callbacks = [tensorboard_callback, modelcp_callback, earlystop_callback]
          
                )

        #Save model weight at end of the training
        #model.save_weights(model_cp_file)

        logger.info(f">>>>>> Training BERT model completed <<<<<<\n\nx==========x")

    else:

        logger.info(f'[training_bert]: Training datafile does not exits: {data_folder}')

#=========================================================================

#import shutil
#shutil.rmtree('/kaggle/working/models') #remove dir with files
#shutil.rmtree('/kaggle/working/logs')

#from IPython.display import FileLink
#FileLink(r'models/checkpoints/logs_hparam_lstm_layers_lr-0_02_lstm_units-544_batch_size-2048_epoch-150/best_model.h5')

#os.listdir('/kaggle/working/models/checkpoints/logs_hparam_lstm_layers_lr-0_2_lstm_units-512_batch_size-2048_epoch-150_')




    