# coding:utf-8
import sys
sys.path.append("../../model")
sys.path.append("../../model/BIMPM/")

import yaml
import json

import pandas as pd
from common import dataloader
from common import evaluator
from common import train_model
from gensim import corpora

import tensorflow as tf
import numpy as np
from math import floor

from common.logger import LoggerHelper
from BIMPM.bimpm_model import BIMPM

def load_config( logger, dataset='quora'):
    dataset = "quora"
    config_path = "../../config/BiMPM_config.yaml"  
    with open(config_path) as fr:
        config = yaml.load(fr)[dataset]
    res = "\n"
    for key,val in config.items():
        res_str = "\n"
        for name,value in val.items():
            res_str += '\t'+ name +" : "+ str(value) + '\n' 
        res += key + " : " + res_str
    logger.info( res )
    return config

def loadData_test( logger, config ):
    train_path = config['data']['train_path']
    valid_path = config['data']['valid_path']
    test_path = config['data']['test_path']
    dictionary_path = config['data']['dictionary_path']
    fixlen = config['training']['fixlen']
    batch_size = config['training']['batch_size']
    pretrain = config['word']['pretrain']
    pretrain_path = config['word']['pretrained_word_path']
    embedding_dim = config['word']['embedding_dim']
    
    logger.debug("Loading data!")
    test_df = pd.read_csv(test_path, sep="\t", nrows=10000)
    dictionary = corpora.Dictionary.load(dictionary_path)
    test_iter = dataloader.load_batched_data_from(test_df,
                                                dictionary=dictionary,
                                                fixlen=fixlen,
                                                batch_size=batch_size)
    return test_iter

def transfor_batchData2FeedDict( batch_data, fixlen ):
    seq1_data = batch_data[1]
    seq2_data = batch_data[2]
    labels = batch_data[3]
    seq_lengths = [ fixlen  for idx in range( len(seq1_data) ) ]  #  最后一个 batch 长度，可能不足 batch_size;
    batch_data_dict = {     'sentence1_word_inputs': np.asarray(seq1_data, dtype=np.int32),
                            'sentence1_word_lengths': np.asarray(seq_lengths, dtype=np.int32),
                            'sentence2_word_inputs': np.asarray(seq2_data, dtype=np.int32),
                            'sentence2_word_lengths': np.asarray(seq_lengths, dtype=np.int32),
                            'labels': np.asarray(labels, dtype=np.int32)
                            # 'is_training':True
                      }
    tmp_length1 = len( np.asarray( seq1_data ) )
    tmp_length2 = len( np.asarray( seq_lengths ) )
    assert  tmp_length1==tmp_length2 , " seq1_data len:{} not match seq_lengths len:{}...".format( tmp_length1, tmp_length2 )
    return batch_data_dict


def eval_validation( true_label, pred_label ):
    assert len(true_label) == len(pred_label), "true_label length != pred_label length."
    tp, tn, fp, fn = 0, 0, 0, 0
    for idx in range( len(true_label) ):
        if true_label[idx] ==1:
            if pred_label[idx] ==1:
                tp +=1
            elif pred_label[idx] ==0:
                fn +=1
        elif true_label[idx] ==0:
            if pred_label[idx] ==1:
                fp +=1
            elif pred_label[idx] ==0:
                tn +=1
    precesion = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = ( 2*tp )/( 2*tp + fp + fn )
    acc = (tp+tn)/len(true_label)
    return acc, f1,recall, precesion, tp, fp, tn, fn


def test_pred( config, test_iter, logger, argv=None):
    fixlen = config['training']['fixlen']
    checkpoint_path = config['data']['checkpoint_path']
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph( checkpoint_path + 'my_test_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint( checkpoint_path ))
        inference = tf.get_collection('run_dict')[1]
        feed_dict = tf.get_collection( "feed_dict" )
        # test
        true_label = []
        pred_label = []
        run_dict = { 'inference':inference }
        for batch_data in test_iter.__iter__():
            batch_data_dict = transfor_batchData2FeedDict( batch_data, fixlen )
            
            run_results = sess.run( run_dict, feed_dict={ 
                                                feed_dict[0]: batch_data_dict['sentence1_word_inputs'],
                                                feed_dict[1]: batch_data_dict['sentence1_word_lengths'],
                                                feed_dict[2]: batch_data_dict['sentence1_word_inputs'],
                                                feed_dict[3]: batch_data_dict['sentence1_word_lengths'],
                                                feed_dict[4]: batch_data_dict['labels'],
                                                feed_dict[5]: False
                                            } )
            # logger.debug( inference )
            pred_label.append( run_results['inference'] )
            true_label.append( batch_data_dict['labels'] )
        
        pred_label = np.hstack( pred_label )
        true_label = np.hstack( true_label )
        acc, f1, recall, precesion, tp, fp, tn, fn = eval_validation( true_label, pred_label )
        logger.info( "Test: acc:{:2.4f}, f1:{:2.4f}, rec:{:2.4f}, prec:{:2.4f}, tp:{}, fp:{}, tn:{}, fn:{}"\
                    .format( acc, f1, recall, precesion, tp, fp, tn, fn ) )
    sess.close()



if __name__=='__main__':
    logHelper = LoggerHelper( log_path='./', log_name='decomAtte_train_demo.log' )
    logger = logHelper.logger
    config = load_config( logger, dataset='quora' )
    test_iter = loadData_test( logger, config )
    test_pred( config, test_iter, logger, argv=None)

