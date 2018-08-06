# coding:utf-8
import json
import yaml
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.rnn import GRUCell

import os
from config_project_rootDir import get_project_rootDir as ProjectRootDir
ProjectRootDIR = ProjectRootDir()

import sys
sys.path.append( '../' )
sys.path.append( os.path.join( ProjectRootDIR, 'Code_for_model/DecomAtten/src/' )  )


import ops.decomposable_attention_ops as decom_ops
from base_model import BaseModel



class DecomposableAttentionClassificationModel(BaseModel):
    """
    Tensorflow model of sentence pair classification
    with decomposable attention model (A Decomposable Attention
    Model for Natural Language Inference, 2016) with
    bi-directional LSTM encoder.
    """

    def __init__(self, config, word_embedding_matrix ):
        super(DecomposableAttentionClassificationModel, self).__init__(config, word_embedding_matrix )
        
    # 定义计算流图 
    def _build_graph(self):
        with tf.name_scope('inputs'):
            sentence1, sentence1_lengths, sentence2, \
                sentence2_lengths, labels, is_training = self._build_inputs()

        
        sentence1_embed, sentence2_embed, sentence1_rnned, sentence2_rnned = \
            self._build_rnn_encoder( sentence1, sentence2,
                                     sentence1_lengths, sentence2_lengths
                                    )
        
        with tf.name_scope('attend'):
            sentence1_attend, sentence2_attend, att_weights1, att_weights2 = \
                decom_ops.attend( sentence1_rnned, sentence2_rnned,
                                  sentence1_lengths, sentence2_lengths,
                                  is_training=is_training
                                )

        with tf.name_scope('compare'):
            compare1, compare2 = decom_ops.compare(
                sentence1_embed, sentence2_embed,
                sentence1_attend, sentence2_attend, is_training=is_training)

        compare_dim = self.config['rnn']['state_size'] * 2
        num_category = self.config['data']['num_category']

        with tf.name_scope('aggregate'):
            result = decom_ops.aggregate(
                compare1, compare2,
                sentence1_lengths, sentence2_lengths,
                mapper_num_layers=[ compare_dim//2, num_category ],
                is_training=is_training
            )
            self.outputs = result
            
        with tf.name_scope('summary_viz'):
            attentions = [
                (att_weights1, 'attention_weight1', sentence2_lengths),
                (att_weights2, 'attention_weight2', sentence1_lengths)
            ]
            for attention_info in attentions:
                self._build_attention_viz( *attention_info )

        self.loss = self._build_loss(result, labels)

        self.inference_probs = tf.nn.softmax( result )
        self.inference = tf.argmax(self.inference_probs, axis=-1, name='inference')
        
        self.train_step, self.train_op = self._build_train_step(self.loss)
        self.summary_op = tf.summary.merge_all()
        
        tf.add_to_collection( "feed_dict", sentence1 )
        tf.add_to_collection( "feed_dict", sentence1_lengths )
        tf.add_to_collection( "feed_dict", sentence2 )
        tf.add_to_collection( "feed_dict", sentence2_lengths )
        tf.add_to_collection( "feed_dict", labels )
        tf.add_to_collection( "feed_dict", is_training )

        tf.add_to_collection( 'pred_network', result )
        tf.add_to_collection( 'pred_network', self.inference )
        tf.add_to_collection( 'pred_network', self.inf )

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

    def _build_rnn_encoder(self, sentence1, sentence2,
                           sentence1_lengths, sentence2_lengths ):

        with tf.variable_scope('word_embedding'):
            sentence1_embedding = tf.nn.embedding_lookup( self._word_embedding, sentence1)
            sentence2_embedding = tf.nn.embedding_lookup( self._word_embedding, sentence2)

        with tf.variable_scope('rnn'):
            def _run_birnn(fw_cell, bw_cell, inputs, lengths):
                (fw_output, bw_output), (fw_final_state, bw_final_state) =\
                    tf.nn.bidirectional_dynamic_rnn(  fw_cell, bw_cell,
                                                      inputs,
                                                      sequence_length=lengths,
                                                      time_major=False,
                                                      dtype=tf.float32   )
                output = tf.concat( [fw_output, bw_output], 2)
                
                state = tf.concat( [fw_final_state, bw_final_state], 1)
                return output, state

            state_size = self.config['rnn']['state_size']
            forward_cell = GRUCell(state_size)
            backward_cell = GRUCell(state_size)

            sentence1_rnned, _ = _run_birnn(forward_cell, backward_cell,
                                            sentence1_embedding, sentence1_lengths)
            sentence2_rnned, _ = _run_birnn(forward_cell, backward_cell,
                                            sentence2_embedding, sentence2_lengths)

        return sentence1_embedding, sentence2_embedding, \
               sentence1_rnned, sentence2_rnned

    def _build_loss(self, logits, labels):
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

    def _build_train_step(self, loss):
        with tf.name_scope('train'):
            train_step = tf.Variable(0, name='global_step', trainable=False)
            lr = self.config['training']['learning_rate']
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            # Returns all variables created with trainable=True.
            train_variables = tf.trainable_variables()  
            grads_vars = opt.compute_gradients(loss, train_variables)  # 根据 loss 计算所有 variables(可更新参数) 的梯度；
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)  #  对每一个参数的梯度进行规约裁剪。控制梯度最大范式，防止梯度爆炸；
            # Apply gradients to variables.
            apply_gradient_op = opt.apply_gradients(grads_vars, global_step=train_step)
            with tf.control_dependencies([ apply_gradient_op ]):
                train_op = tf.no_op(name='train_step')
        return train_step, train_op

    @staticmethod
    def _build_attention_viz(att_weight, att_name, lengths):
        mask = tf.expand_dims(
            tf.sequence_mask(lengths,
                             maxlen=tf.shape(att_weight)[1],
                             dtype=tf.float32),
            axis=-1)
        att_weight = att_weight * mask

        tf.summary.histogram(att_name, att_weight)
        tf.summary.image(att_name + '_viz',
                         tf.cast(
                             tf.expand_dims(att_weight, -1) * 255.0,
                             dtype=tf.uint8))
    