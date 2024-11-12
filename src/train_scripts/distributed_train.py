import tensorflow as tf
from src import logger


def define_distributed_strategy():

    logger.info('[define_distributed_strategy]: Detect TPU or multi GPU, if yes return appropriate distribution strategy...')

    is_TPU_instance_Init = False
    is_Multiple_GPU_instance_Init = False

    num_replicas_in_sync = 1

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
        logger.info('[define_distributed_strategy]: Running on TPU ', tpu.master())
        is_TPU_instance_Init = True
        
    except ValueError:
        tpu = None

    if tpu:
        
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        
        num_replicas_in_sync = strategy.num_replicas_in_sync
        logger.info("[define_distributed_strategy]: REPLICAS: ", strategy.num_replicas_in_sync)
        
    else: #Check for multiple GPU
        
        logger.info('[define_distributed_strategy]: Running on GPU ')

        #Setting for multipl GPU https://towardsdatascience.com/train-a-neural-network-on-multi-gpu-with-tensorflow-42fa5f51b8af
        #to see the list of available GPU devices doing the following
        devices = tf.config.experimental.list_physical_devices('GPU')
        num_replicas_in_sync = len(devices)
        
        logger.info("[define_distributed_strategy]: REPLICAS: ", num_replicas_in_sync)

        if num_replicas_in_sync > 1:
            is_Multiple_GPU_instance_Init = True
            
        #Detect multiple GPU then distribute the task on multiple machine
        #strategy = tf.distribute.MirroredStrategy() #To Supress the warning duing run https://github.com/tensorflow/tensorflow/issues/42146
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        options = tf.data.Options()    
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    if ((is_Multiple_GPU_instance_Init == False) & (is_TPU_instance_Init == False)):
        
        strategy = tf.distribute.get_strategy() 
        num_replicas_in_sync = 1
        logger.info('[define_distributed_strategy]: General strategy...')

    return strategy, num_replicas_in_sync