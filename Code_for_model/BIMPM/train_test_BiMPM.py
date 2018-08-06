# coding:utf-8
import sys
sys.path.append("../../")

import yaml
import json

import numpy as np
import tensorflow as tf

from Code_for_utilTools.logger import LoggerHelper
from src.bimpm_model import BIMPM

from Code_for_data.prep_data  import PrepData
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



def load_config( logger, dataset='quora'):
    dataset = "quora"
    config_path = "./config/BiMPM_config.yaml"  
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
# end  

def transfor_batchData2FeedDict( batch_data, fixlen ):
    seq1_data = batch_data[1]
    seq2_data = batch_data[2]
    labels = batch_data[3]
    seq_lengths = [ fixlen  for idx in range( len(seq1_data) ) ]  #  最后一个 batch 长度，可能不足 batch_size;
    batch_data_dict = {     'sentence1_inputs': np.asarray(seq1_data, dtype=np.int32),
                            'sentence1_lengths': np.asarray(seq_lengths, dtype=np.int32),
                            'sentence2_inputs': np.asarray(seq2_data, dtype=np.int32),
                            'sentence2_lengths': np.asarray(seq_lengths, dtype=np.int32),
                            'labels': np.asarray(labels, dtype=np.int32)
                            # 'is_training':True
                      }
    tmp_length1 = len( np.asarray( seq1_data ) )
    tmp_length2 = len( np.asarray( seq_lengths ) )
    assert  tmp_length1==tmp_length2 , " seq1_data len:{} not match seq_lengths len:{}...".format( tmp_length1, tmp_length2 )
    return batch_data_dict
# end 

def evaluation( true_label, pred_label ):
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
# end  

def __validation(sess, model, logger, data_iter, epoch, data_type='dev' ):
    # eval 
    true_label = []
    pred_label = []
    inference_probs = []
    for batch_data in data_iter:
        feed_dict = model.make_feed_dict(batch_data, is_training=False) 
        run_dict = {'train_op': model.train_op,
                    'inference': model.inference,
                    'inference_probs': model.inference_probs,
                    'loss': model.loss}
        run_results = sess.run(run_dict, feed_dict )
        pred_label.append( run_results['inference'] )
        true_label.append( batch_data['labels'] )

        inference_probs.append( run_dict['inference_probs'] )

    pred_label = np.hstack( pred_label )
    true_label = np.hstack( true_label )
    acc, f1,recall, precesion, tp, fp, tn, fn = evaluation( true_label, pred_label )
    
    # 
    inference_probs = np.hstack( inference_probs )

    logger.info( "{} Eval: epoch:{}, acc:{:2.4f}, f1:{:2.4f}, rec:{:2.4f}, prec:{:2.4f}, tp:{}, fp:{}, tn:{}, fn:{}"\
                .format( data_type, epoch, acc, f1, recall, precesion, tp, fp, tn, fn ) )
    # sklearn evaluation;
    metrics_acc = accuracy_score( true_label, pred_label )
    metrics_f1  = f1_score( true_label, pred_label )
    metrics_recall = recall_score( true_label, pred_label )
    metrics_precision = precision_score( true_label, pred_label )
    logger.info( "{} metrics: epoch:{}, acc:{:2.4f}, f1:{:2.4f}, rec:{:2.4f}, prec:{:2.4f}"\
                .format( data_type, epoch, metrics_acc, metrics_f1, metrics_recall, metrics_precision ) )
    return (acc, f1,recall, precesion, tp, fp, tn, fn),  ( metrics_acc, metrics_f1, metrics_recall, metrics_precision ), inference_probs
# return (acc, f1,recall, precesion, tp, fp, tn, fn),  ( metrics_acc, metrics_f1, metrics_recall, metrics_precesion )


def main_train( batch_num=None ):
    # init logger 
    logHelper = LoggerHelper()
    logHelper.addFileHanlder( log_path='./log', log_name='BiMPM_train_demo.log' )
    logger = logHelper.logger
    # load config info
    config = load_config( logger, dataset='quora' )
    # prepratation data
    prep_data = PrepData(  dataPath=str('dataSets/quora/'), wordsNum=60000, charsNum=1200  )
    word_dict = prep_data.getDict(  dictName='word_count_dict.txt'  )
    char_dict = prep_data.getDict(  dictName='char_count_dict.txt'  )
    prep_data.filterWordCharDict_saveVoc2idx(  word_dict, char_dict  )
    # load w2v
    word_embedding = prep_data.load_w2v()
    print( "word Num: {}".format( prep_data.wordsNum ) )
    print( "char Num: {}".format( prep_data.charsNum ) )
    config['data']['num_word'] = prep_data.wordsNum

    print( "word_embedding type: {}".format( type(word_embedding) ) )
    prep_data.get_sample_num() # get the number of samples;
    # print( prep_data.sample_num )
    train_smaple_num = prep_data.sample_num['new_train.tsv']
    dev_smaple_num = prep_data.sample_num['new_dev.tsv']
    print( "train sample num: {}; dev sample num: {}".format( train_smaple_num, dev_smaple_num ) )

    
    with_char = config['char']['with_char']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    checkpoint_path = config['data']['checkpoint_path']
    model = BIMPM(config, word_embedding)

    summary_step = 300
    log_step = 300

    with tf.Session() as sess:
        sess.run([ tf.global_variables_initializer(), tf.local_variables_initializer() ])
        # summary_writer = tf.summary.FileWriter( checkpoint_path, sess.graph ) # 将 sess.graph 图写入  train_dir 目录里；
        # log.warning("Training Start!")
        epoch_acc_score = [  ]
        best_eva_tuple = tuple()
        epoch_metrics_acc_score = []
        best_metrics_tuple = tuple()

        inference_probs_epoch = []

        step = 0
        for epoch in range(1, epochs+1):
            train_iter = prep_data.get_BatchData( file_name='new_train.tsv', batch_size= batch_size, 
                                                  with_char_inputs=with_char, batch_num=batch_num, batch_padding=True )
            for batch_data in train_iter:
                feed_dict = model.make_feed_dict( batch_data, is_training=True ) 
                # print( "batch_data: {}".format( batch_data ) )
                for item,val in batch_data.items():
                    print( "{}:{}".format( item, val.shape ) )
                # print( "feed_dict: {}".format( feed_dict ) )
                run_dict = {'train_op': model.train_op,
                            'inference': model.inference,
                            'inference_probs': model.inference_probs,
                            'loss': model.loss}
                if (step + 1) % summary_step == 0:
                    run_dict['summary_op'] = model.summary_op
                run_results = sess.run(run_dict, feed_dict )
                if (step + 1) % log_step == 0:
                    logger.debug("Step {cur_step:6d} (Epoch {float_epoch:6.3f}) ... Loss: {loss:.5f}"
                                .format(cur_step=step+1,
                                        float_epoch=float(step+1)/ (step*2),# steps_in_epoch,
                                        loss=run_results['loss'])  )
                step += 1
            # get data iterator. 
            train_iter = prep_data.get_BatchData( file_name='new_train.tsv', batch_size=batch_size, 
                                                  with_char_inputs=False, batch_num=batch_num, batch_padding=True )
            dev_iter = prep_data.get_BatchData( file_name='new_dev.tsv', batch_size=batch_size, 
                                                with_char_inputs=False, batch_num=batch_num, batch_padding=True )
            # Evaluation on train_data: 
            train_eval_tuple, train_metrics_tuple, _ = __validation( sess, model, logger, train_iter, epoch, data_type='train' )
            # Evaluation on dev_data:
            dev_eval_tuple, dev_metrics_tuple, dev_inference_probs = __validation( sess, model, logger, dev_iter, epoch, data_type='dev' )
            
            if epoch % 10 ==0:
                inference_probs_epoch.append( dev_inference_probs )
                    
            acc, f1,recall, precesion, tp, fp, tn, fn = dev_eval_tuple
            metrics_acc, metrics_f1, metrics_recall, metrics_precesion = dev_metrics_tuple
            # evaluation
            epoch_acc_score.append( acc )
            if max(epoch_acc_score) == acc: # 存储 当前 acc 最优的模型；
                best_eva_tuple = (epoch, acc, f1, recall, precesion, tp, fp, tn, fn)
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path + "my_test_model")
            # metrics
            epoch_metrics_acc_score.append( metrics_acc )
            if max(epoch_metrics_acc_score) == metrics_acc:
                best_metrics_tuple = (epoch, metrics_acc, metrics_f1, metrics_recall, metrics_precesion)
        logger.info( "Eval:  Best Acc:  epoch:{}, acc:{:2.4f}, f1:{:2.4f} rec:{:2.4f}, prec:{:2.4f}, tp:{}, fp:{}, tn:{}, fn:{}"\
                        .format( *best_eva_tuple ) )
        logger.info( "sklearn-metrics:  Best Acc:  epoch:{}, acc:{:2.4f}, f1:{:2.4f} rec:{:2.4f}, prec:{:2.4f}"\
                        .format( *best_metrics_tuple ) )
        
        infer_probs_output_str = ""
        for infer_probs in inference_probs_epoch:
            infer_probs_output_str += '\t'.join( infer_probs ) + '\n'
        with open( 'result_infer_probs_epoch.txt', 'w' ) as output_file:
            output_file.write( infer_probs_output_str )


def test_pred( config, valid_iter, logger, argv=None):
    config['data']['num_word'] = Num_words
    epochs = config['training']['epochs']
    fixlen = config['training']['fixlen']
    checkpoint_path = config['data']['checkpoint_path']

    summary_step = 300
    log_step = 300
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph( checkpoint_path + 'my_test_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint( checkpoint_path ))
        # pred_network_list = tf.get_collection('pred_network')
        inference = tf.get_collection('pred_network')[1]
        inference_probs = tf.get_collection('pred_network')[2]
        print( inference )
        run_dict = {
            'inference': inference,
            'inference_probs': inference_probs
        }

        feed_dict = tf.get_collection( "feed_dict" )
        # eval
        true_label = []
        pred_label = []
        for batch_data in valid_iter.__iter__():
            batch_data_dict = transfor_batchData2FeedDict( batch_data, fixlen )
            
            run_results = sess.run( run_dict, feed_dict={ 
                                                feed_dict[0]: batch_data_dict['sentence1_inputs'],
                                                feed_dict[1]: batch_data_dict['sentence1_lengths'],
                                                feed_dict[2]: batch_data_dict['sentence1_inputs'],
                                                feed_dict[3]: batch_data_dict['sentence1_lengths'],
                                                feed_dict[4]: batch_data_dict['labels'],
                                                feed_dict[5]: False
                                            } )
            pred_label.append( run_dict['inference'] )
            true_label.append( batch_data_dict['labels'] )
        
        pred_label = np.hstack( pred_label )
        true_label = np.hstack( true_label )
        acc, f1, recall, precesion, tp, fp, tn, fn = eval_validation( true_label, pred_label )
        logger.info( "Test:  acc:{:2.4f}, f1:{:2.4f}, rec:{:2.4f}, prec:{:2.4f}, tp:{}, fp:{}, tn:{}, fn:{}"\
                    .format( acc, f1, recall, precesion, tp, fp, tn, fn ) )
    sess.close()



if __name__=='__main__':
    main_train( batch_num=10 )

