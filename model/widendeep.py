
import os
import  tensorflow as tf
from tensorflow.estimator import DNNLinearCombinedClassifier
#from util.utils import timestamp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time


def timestamp():
    return str(int(time.time()))


def wdl(linear_featcols, dnn_featcols, configMap):
    model_dir = configMap['model_dir']
    #model_dir = "/summary_dir/"

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000)# 每隔多少步检查点
    session_config = tf.ConfigProto(device_count={"CPU": os.cpu_count()},
                              inter_op_parallelism_threads=os.cpu_count(),
                              intra_op_parallelism_threads=os.cpu_count(),
                              log_device_placement=True)

    # session_config = tf.ConfigProto(device_count={"CPU": 1},
    #                           inter_op_parallelism_threads=0,
    #                           intra_op_parallelism_threads=0,
    #                           log_device_placement=False)
    run_config = run_config.replace(session_config=session_config)
    # lr = tf.train.exponential_decay(
    #     learning_rate=0.01,
    #     global_step=tf.compat.v1.train.get_global_step(),
    #     decay_steps=5000,
    #     decay_rate=0.96, staircase=False)

    return DNNLinearCombinedClassifier(
        config = run_config,
        dnn_dropout=0.3,
        # wide settings
        linear_feature_columns=linear_featcols,
        ##linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.001, l1_regularization_strength=0.001, l2_regularization_strength=0.001),
        linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.003, 

            l1_regularization_strength=0.01, 
            l2_regularization_strength=0.01
         ),


        # deep settings
        dnn_feature_columns=dnn_featcols,
        dnn_hidden_units=configMap['dnn_hidden_units'],



        dnn_optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01),


        model_dir=model_dir,
        loss_reduction=tf.losses.Reduction.MEAN,

        )



