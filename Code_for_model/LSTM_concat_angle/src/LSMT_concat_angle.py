# coding:utf-8
import numpy
import tensorflow as tf

from base_model import BaseModel

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTM



class LSTM_concat_angle(BaseModel):
    def __init__(self):
        super(LSTM_concat_angle, self).__init__(config, word_embedding_matrix )
    
    def _build_graph(self):

        pass

    def _build_inputs(self):
        self._inputs['s1_words_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='s1_words_inputs'
        )
        self._inputs['s1_words_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='s1_words_lengths'
        )
        self._inputs['s2_words_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='s2_words_inputs'
        )
        self._inputs['s2_words_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='s2_words_lengths'
        )
        self._inputs['labels'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='labels'
        )
        self._inputs['is_training'] = tf.placeholder(
            shape=tuple(),
            dtype=tf.bool,
            name='is_training'
        )
        return self._inputs['s1_words_inputs'], self._inputs['s1_words_lengths'], \
                 self._inputs['s2_words_inputs'], self._inputs['s2_words_lengths'], \
                 self._inputs['labels'], self._inputs['is_training']


    def _build_rnn_encoder(self, s1_words_inputs, s1_words_lengths, s2_words_inputs, s2_words_lengths):
        with tf.variable_scope("word_embedding"):
            s1_embedding = tf.nn.embedding_lookup( self._word_embedding, s1_words_inputs )
            s2_embedding = tf.nn.embedding_lookup( self._word_embedding, s2_words_inputs )
        with tf.variable_scope('rnn_encoder'):
            def _run_birnn( fw_cell, bw_cell, inputs, lengths ):
                (fw_output, bw_output), (fw_final_state, bw_final_state) = \
                    tf.nn.bidirectional_dynamic_rnn(  fw_cell, bw_cell,
                                                      inputs,
                                                      sequence_length=lengths,
                                                      time_major=False,
                                                      dtype=tf.float32   )
                output = tf.concat( [fw_output, bw_output], 2)
                # state = tf.concat( [fw_final_state, bw_final_state], 1)
                return output, fw_final_state, bw_final_state
            # 
            state_size = self.config['rnn']['state_size']

            forward_cell = GRUCell(state_size)
            backward_cell = GRUCell(state_size)
            s1_output, s1_fw_final_state, s1_bw_final_state = _run_birnn( forward_cell, backward_cell, s1_words_inputs, s1_words_lengths )
            s2_output, s2_fw_final_state, s2_bw_final_state = _run_birnn( forward_cell, backward_cell, s2_words_inputs, s2_words_lengths )
        return (s1_output, s1_fw_final_state, s1_bw_final_state), (s2_output, s2_fw_final_state, s2_bw_final_state)



    def _build_loss(self, logit, label):
        with tf.name_scope('loss'):
            onehot_labels = tf.one_hot(labels,
                                       depth=self.config['data']['num_category'],
                                       dtype=tf.int32)
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=onehot_labels,
                    logits=logits,
                    name='cross_entropy'
                )
            )
            # 添加了 l2 正则；
            l2_loss = tf.add_n(  tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES )  )
            loss = cross_entropy + l2_loss 

            # 作用: 输出一个包含单个标量值的summary协议缓冲区.
            tf.summary.scalar('total_loss', loss)
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('l2_loss', l2_loss)
        return loss





        pass

    def _build_train_op(self, loss):
        
        pass
