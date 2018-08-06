# coding:utf-8
import sys
sys.path.append( '../../' )

import numpy as np
import tensorflow as tf

from Code_for_model.BIMPM.src.base_model import BaseModel
import Code_for_model.BIMPM.src.util.layer_utils as layer_utils
import Code_for_model.BIMPM.src.util.match_utils as match_utils
# import util.my_rnn as my_rnn
import Code_for_model.BIMPM.src.util.mlp_layers as mlp_layers


class BIMPM(BaseModel):
    """
    Tensorflow model of sentence pair classification
    with BiMPM;
    """
    def __init__(self, config, word_embedding_matrix=None ):
        super(BIMPM, self).__init__(config, word_embedding_matrix)
        
    def _build_graph(self):
        char_inputs_tuple = None # 初始化 char_inputs_tuple
        labels, is_training, s1_word_inputs, s1_word_lengths, s2_word_inputs, s2_word_lengths = self._build_word_inputs()
        
        # add word_inputs into feed_dict
        tf.add_to_collection( "feed_dict", s1_word_inputs )
        tf.add_to_collection( "feed_dict", s1_word_lengths )
        tf.add_to_collection( "feed_dict", s2_word_inputs )
        tf.add_to_collection( "feed_dict", s2_word_lengths )
        tf.add_to_collection( "feed_dict", labels )
        tf.add_to_collection( "feed_dict", is_training )

        if self.config['char']['with_char']:
            s1_char_inputs, s1_char_lengths, s2_char_inputs, s2_char_lengths = self._build_char_inputs()
            char_inputs_tuple = ( s1_char_inputs, s1_char_lengths, s2_char_inputs, s2_char_lengths )
            # add char_inputs into feed_dict
            tf.add_to_collection( "feed_dict", s1_char_inputs )
            tf.add_to_collection( "feed_dict", s1_char_lengths )
            tf.add_to_collection( "feed_dict", s2_char_inputs )
            tf.add_to_collection( "feed_dict", s2_char_lengths )
            with tf.variable_scope('char_embedding'):
                embedding_kwargs = {}
                initializer = tf.contrib.layers.xavier_initializer()
                embedding_kwargs['shape'] = (self.config['char']['num_char'], self.config['char']['embedding_dim'])
                self._char_embedding = tf.get_variable( name="char_embedding", initializer=initializer, dtype=tf.float32, **embedding_kwargs )

        s1_repres, s2_repres, input_dim, s1_words_mxlen, s2_words_mxlen = self._build_word_representation_layer( s1_word_inputs, s2_word_inputs,
                                                                                                                 s1_word_lengths, s2_word_lengths, 
                                                                                                                 char_inputs_tuple=char_inputs_tuple  )

        if is_training is not None:
            s1_repres = tf.nn.dropout(s1_repres, (1 - self.config['model']['dropout_rate'] ))
            s2_repres = tf.nn.dropout(s2_repres, (1 - self.config['model']['dropout_rate'] ))
        
        mx_words_len = np.max( s1_words_mxlen, s2_words_mxlen )
        passages_mask = tf.sequence_mask( s1_word_lengths, mx_words_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask( s2_word_lengths, mx_words_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if self.config['model']['with_highway']:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer( s1_repres, input_dim, self.config['model']['highway_layer_num'] )
                tf.get_variable_scope().reuse_variables()
                in_passages_repres = match_utils.multi_highway_layer( s2_repres, input_dim, self.config['model']['highway_layer_num'] )

        # ========Bilateral Matching=====
        config = self.config
        (match_representation, match_dim) = match_utils.bilateral_match_func( in_question_repres, in_passages_repres,
                                                                              s1_word_lengths, s2_word_lengths, 
                                                                              question_mask, passages_mask,  
                                                                              input_dim, is_training, 
                                                                              config=config )
        #========Prediction Layer=========
        # match_dim = 4 * self.config['aggregation_lstm_dim']
        mapper_l2_coef=3e-4
        mapper_num_layers = [ match_dim//2, self.config['data']['num_category'] ]
        mapper = mlp_layers.MLP(
            mapper_num_layers,
            dropout=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=mapper_l2_coef),
            name='aggregate_mapper',
        )
        logits = mapper.apply( match_representation, is_training=is_training )
        # 
        self.loss = self._build_loss(logits, labels)
        self.inference_probs = tf.nn.softmax( logits, name='inference_probs' )
        self.inference = tf.argmax(self.inference_probs, axis=-1, name='inference')
        
        self.train_step, self.train_op = self._build_train_step(self.loss)
        self.summary_op = tf.summary.merge_all()
        tf.add_to_collection( "pred_network", self.loss )
        tf.add_to_collection( "pred_network", self.inference )
        tf.add_to_collection( "pred_network", self.inference_probs )
        tf.add_to_collection( "pred_network", self.train_op )
        

    def _build_word_inputs(self):
        self._inputs['labels'] = tf.placeholder( shape=(None,), dtype=tf.int32, name='labels' )
        self._inputs['is_training'] = tf.placeholder( shape=tuple(), dtype=tf.bool, name='is_training' )
        self._inputs['s1_words_inputs'] = tf.placeholder( shape=(None, None),  dtype=tf.int32, name='s1_words_inputs' )# batch_size, max_time
        self._inputs['s1_words_lengths'] = tf.placeholder( shape=(None,), dtype=tf.int32, name='s1_words_lengths' )
        self._inputs['s2_words_inputs'] = tf.placeholder( shape=(None, None),  dtype=tf.int32, name='s2_words_inputs' )# batch_size, max_time
        self._inputs['s2_words_lengths'] = tf.placeholder( shape=(None,), dtype=tf.int32, name='s2_words_lengths' )
        return self._inputs['labels'], self._inputs['is_training'],\
                self._inputs['s1_words_inputs'], self._inputs['s1_words_lengths'],\
                self._inputs['s2_words_inputs'], self._inputs['s2_words_lengths']
        
    def _build_char_inputs(self):
        if self.config['with_char']:
            self._inputs['s1_chars_lengths'] = tf.placeholder( shape=(None,None), dtype=tf.int32, name='s1_chars_lengths' ) # [batch_size, question_len]
            self._inputs['s2_chars_lengths'] = tf.placeholder( shape=(None,None), dtype=tf.int32, name='s2_chars_lengths' ) # [batch_size, passage_len]
            self._inputs['s1_chars_inputs'] = tf.placeholder( shape=(None,None,None), dtype=tf.int32, name='s1_chars_inputs' ) # [batch_size, question_len, sentence1_char_len]
            self._inputs['s2_chars_inputs'] = tf.placeholder( shape=(None,None,None), dtype=tf.int32, name='s2_chars_inputs' ) # [batch_size, passage_len, p_char_len]
            return self._inputs['s1_chars_lengths'], self._inputs['s2_chars_lengths'],\
                self._inputs['s1_chars_inputs'], self._inputs['s2_chars_inputs']
        

    # ====== word representation layer ======
    def _build_word_representation_layer(self, s1_words_inputs, s2_words_inputs, 
                                               s1_words_lengths, s2_words_lengths, 
                                               char_inputs_tuple=None ):
        # 存放 word  和 char 的结果；
        in_question_repres = []   
        in_passage_repres = []
        input_dim = 0
        
        # print( "-----------s2_words_inputs: {}".format( s2_words_inputs ) )
        s1_word_repres = tf.nn.embedding_lookup(self._word_embedding, s1_words_inputs ) # [batch_size, question_len, word_dim]
        s2_word_repres = tf.nn.embedding_lookup(self._word_embedding, s2_words_inputs ) # [batch_size, passage_len, word_dim]
        in_question_repres.append( s1_word_repres )
        in_passage_repres.append( s2_word_repres )
        # print( "-----------s2_word_repres: {}".format( s2_word_repres ) )

        # input_shape = tf.shape( s1_words_inputs )
        input_shape = s1_word_repres.get_shape().as_list()
        print( "-----------input_shape: {}".format( input_shape ) )
        batch_size = input_shape[0]
        self.s1_words_mxlen = input_shape[1]  # 获取 每个 batch_data 的 s1_word_mxlen，即 s1 输入的 词总数；
        
        # input_shape = tf.shape( s2_words_inputs )
        input_shape = s2_word_repres.get_shape().as_list()
        print( "-----------input_shape: {}".format( input_shape ) )
        self.s2_words_mxlen = input_shape[1]  # 获取 每个 batch_data 的 s2_word_mxlen，即 s1 输入的 词总数；
        input_dim += self.config['word']['embedding_dim']
        
        if self.config['char']['with_char'] and char_inputs_tuple is not None:
            s1_chars_inputs, s2_chars_inputs, s1_chars_lengths, s2_chars_lengths = char_inputs_tuple

            input_shape = tf.shape( s1_chars_inputs )
            batch_size = input_shape[0]
            s1_words_mxlen = input_shape[1]
            q_char_len = input_shape[2]

            input_shape = tf.shape( s2_chars_inputs )
            s2_words_mxlen = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = self.config['char']['embedding_dim']
            # 
            in_question_char_repres = tf.nn.embedding_lookup(self._char_embedding, s1_chars_inputs ) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])  # [ batch_size * question_len,  q_char_len, char_dim] 压平
            question_char_lengths = tf.reshape( s1_chars_lengths, [-1])
            quesiton_char_mask = tf.sequence_mask(question_char_lengths, q_char_len, dtype=tf.float32)  # [ batch_size * question_len, q_char_len]
            in_question_char_repres = tf.multiply(in_question_char_repres, tf.expand_dims(quesiton_char_mask, axis=-1))
            # 
            in_passage_char_repres = tf.nn.embedding_lookup(self._char_embedding, s2_chars_inputs ) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape( s2_chars_lengths, [-1])
            passage_char_mask = tf.sequence_mask(passage_char_lengths, p_char_len, dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
            in_passage_char_repres = tf.multiply(in_passage_char_repres, tf.expand_dims(passage_char_mask, axis=-1))
            # 
            (question_char_outputs_fw, question_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_question_char_repres, 
                                                                                                self.config['char_lstm_dim'],
                                                                                                input_lengths=s1_chars_lengths,
                                                                                                scope_name="char_lstm", 
                                                                                                reuse=False,
                                                                                                is_training=self.config['is_training'], 
                                                                                                dropout_rate=self.config['dropout_rate'], 
                                                                                                use_cudnn=self.config['use_cudnn'] )
            # 
            question_char_outputs_fw = layer_utils.collect_final_step_of_lstm(question_char_outputs_fw, question_char_lengths - 1)
            question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
            question_char_outputs = tf.concat( axis=1, values=[question_char_outputs_fw, question_char_outputs_bw] )
            question_char_outputs = tf.reshape( question_char_outputs, [batch_size, s1_words_mxlen, 2*self.config['char_lstm_dim'] ] )
            # 
            (passage_char_outputs_fw, passage_char_outputs_bw, _) = layer_utils.my_lstm_layer(  in_passage_char_repres, 
                                                                                                self.config['char_lstm_dim'],
                                                                                                input_lengths=s2_chars_lengths,
                                                                                                scope_name="char_lstm", 
                                                                                                reuse=False,
                                                                                                is_training=self.config['is_training'], 
                                                                                                dropout_rate=self.config['dropout_rate'], 
                                                                                                use_cudnn=self.config['use_cudnn'] )
            # 
            passage_char_outputs_fw = layer_utils.collect_final_step_of_lstm(passage_char_outputs_fw, passage_char_lengths - 1)
            passage_char_outputs_bw = passage_char_outputs_bw[:, 0, :]
            passage_char_outputs = tf.concat(axis=1, values=[passage_char_outputs_fw, passage_char_outputs_bw])
            passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, s2_words_mxlen, 2*self.config['char_lstm_dim'] ] )

            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)
            input_dim += 2*self.config['char_lstm_dim']

        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]
        return in_question_repres, in_passage_repres, input_dim, self.s2_words_mxlen, self.s2_words_mxlen
    

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

