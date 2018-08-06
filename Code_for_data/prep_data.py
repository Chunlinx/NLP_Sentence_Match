# coding:utf-8
import sys
sys.path.append( '../' )
import os
# from config_project_rootDir import get_project_rootDir as ProjectRootDir
# ProjectRootDIR = ProjectRootDir()
ProjectRootDIR = "/Users/gaoyong/work_job/neeson_github/Code_Notes/"


import time


from Code_for_utilTools.logger import LoggerHelper

import numpy as np
import csv
import json
import logging

import pickle
import collections

from gensim.models import Word2Vec

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# The data iterator for word2vec to train model ...  
class SentenceIters():
    def __init__(self, dataPath='dataSets/quora/', 
                       file_name_list=[ 'new_train.tsv', 'new_dev.tsv', 'new_test.tsv' ] ):
        self.dataPath = dataPath
        self.file_name_list = file_name_list
    def __iter__(self):
        for file_name in self.file_name_list:
            absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + file_name )
            with open( absolute_dataPath, 'r' ) as csvfile:
                csvReader = csv.DictReader(csvfile, delimiter='\t')
                for row in csvReader:
                    yield row['q1'].strip().split()
                    yield row['q2'].strip().split()
    # func end

# class end


class PrepData():
    def __init__( self, dataPath=str('dataSets/quora/'), wordsNum=50000, charsNum=1200 ):
        self.num_category = 2
        self.sample_num = collections.defaultdict(int)

        self.dataPath = dataPath
        logHelper = LoggerHelper()
        self.logger = logHelper.logger

        self._word_idx2vocab = ['PAD', 'UNK']
        self._char_idx2vocab = ['PAD', 'UNK']
        self.PAD = 0
        self.UNK = 1
        self.max_char_len_per_word = 10
        self._word_vocab2idx = {}
        self._char_vocab2idx = {}
        self.wordsNum = wordsNum
        self.charsNum = charsNum
        print( "init_done! " ) 
    
    # get the BatchData;  a generator object
    def get_BatchData(self, file_name, batch_size=128, with_char_inputs=False, batch_num=None, batch_padding=False,
                                       s1_key='q1', s2_key='q2', label_key='label'):
        # 
        file_sample_num = self.sample_num[file_name]
        if batch_num is not None:
            assert (batch_num * batch_size) <= file_sample_num, " The batch_num * batch_size:{}*{} is out of range, the sample_num: {}. "\
                                                                .format( batch_num, batch_size, file_sample_num )
        # 只利用 yield 对大文件，进行切片读取。逐行读取，便于对 每行 进行预处理（包括分词，去停用词，各行 分列 处理等 ）。
        def _normalize_length(_data, max_length):
            if max_length >= len(_data):
                return  [ self.PAD ] * ( max_length - len(_data)) + _data
            else:
                return _data[:max_length]
        s1_words_inputs, s1_words_lengths, s2_words_inputs, s2_words_lengths, labels = [], [], [], [], []
        s1_chars_inputs, s2_chars_inputs = [], []
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + file_name )
        with open( absolute_dataPath, 'r' ) as csvfile:
            csvReader = csv.DictReader(csvfile, delimiter='\t')
            batch_num_idx = 0
            index = 0
            for row in csvReader:
                index +=1
                # self.__encoder_wordsSeq( seq_str ) # transfer the seq_str to label_list( like a Interger list )
                q1_seq, q2_seq, label = self.__encoder_wordsSeq(row[ s1_key ]), self.__encoder_wordsSeq(row[ s2_key ]), row[ label_key ]
                s1_words_inputs.append( q1_seq )
                s1_words_lengths.append( len(q1_seq) )
                s2_words_inputs.append( q2_seq )
                s2_words_lengths.append( len(q2_seq) )
                labels.append( label )
                if with_char_inputs==True:
                    s1_char_seq, s2_char_seq = self.__encoder_charsSeq(row[ s1_key ]), self.__encoder_charsSeq(row[ s2_key ])
                    s1_char_seq = list(map( lambda item: _normalize_length(item, self.max_char_len_per_word), s1_char_seq ))
                    s2_char_seq = list(map( lambda item: _normalize_length(item, self.max_char_len_per_word), s2_char_seq ))
                    s1_chars_inputs.append( s1_char_seq )
                    s2_chars_inputs.append( s2_char_seq )
                if index % batch_size ==0:
                    seq1_max_length = max(s1_words_lengths)
                    seq2_max_length = max(s2_words_lengths)
                    if batch_padding== True:
                        max_len = max( seq1_max_length, seq2_max_length )
                        seq1_max_length = max_len
                        seq2_max_length = max_len
                        s1_words_lengths = [ max_len ] * batch_size
                        s2_words_lengths = [ max_len ] * batch_size
                    s1_words_inputs = list(map( lambda item: _normalize_length(item, seq1_max_length), s1_words_inputs ) ) 
                    s2_words_inputs = list(map( lambda item: _normalize_length(item, seq2_max_length), s2_words_inputs ) )
                    # seq1_data = [ list(map(int, item_list)) for item_list in seq1_data ]
                    # seq2_data = [ list(map(int, item_list)) for item_list in seq2_data ]
                    batch_data_dict = {     's1_words_inputs':  np.asarray( s1_words_inputs,  dtype=np.int32),
                                            's1_words_lengths': np.asarray( s1_words_lengths, dtype=np.int32),
                                            's2_words_inputs':  np.asarray( s2_words_inputs,  dtype=np.int32),
                                            's2_words_lengths': np.asarray( s2_words_lengths, dtype=np.int32),
                                            'labels': np.asarray(labels, dtype=np.int32)
                                      }
                    if with_char_inputs==True:
                        batch_data_dict['s1_chars_inputs'] =   s1_chars_inputs 
                        batch_data_dict['s1_chars_lengths'] =  s1_words_lengths 
                        batch_data_dict['s1_chars_inputs'] =   s2_chars_inputs 
                        batch_data_dict['s1_chars_lengths'] =  s2_words_lengths 
                    yield batch_data_dict
                    s1_words_inputs, s1_words_lengths, s2_words_inputs, s2_words_lengths, labels = [], [], [], [], []
                    s1_chars_inputs, s2_chars_inputs = [], []
                    # sample a small datasets ; 
                    batch_num_idx +=1
                    if batch_num is not None:
                        if batch_num_idx >= batch_num:
                            return batch_data_dict
            if len( labels ) != 0:
                seq1_max_length = max(s1_words_lengths)
                seq2_max_length = max(s2_words_lengths)
                if batch_padding== True:
                    max_len = max( seq1_max_length, seq2_max_length )
                    seq1_max_length = max_len
                    seq2_max_length = max_len
                    s1_words_lengths = [ max_len ] * batch_size
                    s2_words_lengths = [ max_len ] * batch_size
                s1_words_inputs = list( map( lambda item: _normalize_length(item, seq1_max_length), s1_words_inputs ) )
                s2_words_inputs = list( map( lambda item: _normalize_length(item, seq2_max_length), s2_words_inputs ) )
                
                batch_data_dict = {     's1_words_inputs':  np.asarray(s1_words_inputs,  dtype=np.int32),
                                        's1_words_lengths': np.asarray(s1_words_lengths, dtype=np.int32),
                                        's2_words_inputs':  np.asarray(s2_words_inputs,  dtype=np.int32),
                                        's2_words_lengths': np.asarray(s2_words_lengths, dtype=np.int32),
                                        'labels': np.asarray(labels, dtype=np.int32)
                                    }
                if with_char_inputs==True:
                        batch_data_dict['s1_chars_inputs'] =  np.asarray( s1_chars_inputs,  dtype=np.int32 )
                        batch_data_dict['s1_chars_lengths'] = np.asarray( s1_words_lengths, dtype=np.int32 )
                        batch_data_dict['s1_chars_inputs'] =  np.asarray( s2_chars_inputs,  dtype=np.int32 )
                        batch_data_dict['s1_chars_lengths'] = np.asarray( s2_words_lengths, dtype=np.int32 )
                yield batch_data_dict
        # end
    # end


    # encoder wordSeq to idxSeq
    def __encoder_wordsSeq(self, seq_str):  # transfer word-sequence to idx-sequence
        words = seq_str.split()
        return [ int(self._word_vocab2idx[word]) if word in self._word_vocab2idx else self.UNK for word in words]
    # end

    # encoder charSeq to idxSeq
    def __encoder_charsSeq(self, seq_str):
        charSeq = []
        # print( "__encoder_charsSeq: ", seq_str )
        for word in seq_str.split():
            word_idx = [ int(self._char_vocab2idx[char] ) if char in self._char_vocab2idx else self.UNK for char in word ]
            charSeq.append( word_idx )
        # print( " ######## __encoder_charsSeq: ", charSeq )
        return  charSeq
    # end
    
    # read file_list, and get the word_count_dict, char_count_dict;
    def save_Word_Char_Dict(self, file_name_list, s1_key='q1', s2_key='q2', label_key='label' ):
        
        word_count_dict = collections.defaultdict(int)
        char_count_dict = collections.defaultdict(int)
        label_dict = collections.defaultdict(int)
        # read data by dataPath element
        for file_name in file_name_list:
            absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + file_name )
            with open( absolute_dataPath, 'r' ) as csvfile:
                csvReader = csv.DictReader(csvfile, delimiter='\t')
                sample_num = 0
                for row in csvReader:
                    label = row[ label_key ]
                    label_dict[ label ] +=1
                    sample_num +=1
                    q1_list = row[ s1_key ].strip().split()
                    q2_list = row[ s2_key ].strip().split()
                    for word in q1_list + q2_list:
                        word_count_dict[word] +=1
                    for char in row[ s1_key ] + row[ s2_key ]:
                        char_count_dict[char] +=1
            self.logger.debug( "From:{}, Have read {} samples.".format( file_name, sample_num ) )
            self.sample_num[ file_name ] = sample_num
        self.num_category = len( label_dict )
        self.logger.debug( "read and write done...  num_category size:{}".format( self.num_category ) )
        # sorted the word_count_dict, char_count_dict;
        word_count_dict = { key:val for key,val in sorted( word_count_dict.items(), key=lambda x: (x[1],x[0]), reverse=True ) }
        char_count_dict = { key:val for key,val in sorted( char_count_dict.items(), key=lambda x: (x[1],x[0]), reverse=True ) }
        # save the dict
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + "word_count_dict.txt")
        with open( absolute_dataPath, 'w' ) as dictFile:
            for key,val in word_count_dict.items():
                dictFile.write( str(key) +" "+str(val) +"\n" )
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + "char_count_dict.txt")
        with open( absolute_dataPath, 'w' ) as dictFile:
            for key,val in char_count_dict.items():
                dictFile.write( str(key) +" "+str(val) +"\n" )
        self.logger.debug( "save word_count_dict done... Path:{}; word_count_dict size:{}".format( self.dataPath + "word_count_dict.txt", len(word_count_dict) ) )
        self.logger.debug( "save char_count_dict done... Path:{}; char_count_dict size:{}".format( self.dataPath + "char_count_dict.txt", len(char_count_dict) ) )
        
        # save the dataSet describe file ...
        sample_num_dict = self.sample_num
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + "dataSet_describe.txt" )
        with open( absolute_dataPath, 'w' ) as describe_file:
            describe_file.write( json.dumps(sample_num_dict) )

        return self.sample_num, word_count_dict, char_count_dict
    # end

    # filter low-frequence word, and save char2idx_dict, word2idx_dict;
    def filterWordCharDict_saveVoc2idx(self, word_dict, char_dict, lowFreq=1 ):
        # filter lowFreq word
        new_wordDict = {  }
        for word,count in word_dict.items():
            if count <= lowFreq:
                continue
            new_wordDict[word] = count
        # for word_dict
        new_wordDict_list = sorted(new_wordDict.items(), key=lambda x: (x[1], x[0]), reverse=True) # reverse sorted, from big to small
        self._word_idx2vocab.extend( [word for word,count in new_wordDict_list] )  # using wordDict  extend  idx2vocab list.
        if self.wordsNum is not None:  #  self.wordsNum is not None, filter the low freq word.
            if len(self._word_idx2vocab) >= self.wordsNum:
                self._word_idx2vocab = self._word_idx2vocab[: self.wordsNum]
            else:
                self.wordsNum = len( self._word_idx2vocab )
        self._word_vocab2idx = {v:i for i,v in enumerate(self._word_idx2vocab)}  # get idx of the word
        self.logger.debug( "word_dict size:{}; words num:{}.".format( len(self._word_vocab2idx), self.wordsNum ) )
        # for char_dict
        new_charDict_list = sorted( char_dict.items(), key=lambda x: (x[1], x[0]), reverse=True )
        self._char_idx2vocab.extend( [ char for char,count in new_charDict_list ] )
        if self.charsNum is not None:  #  self.charsNum is not None, filter the low freq char.
            if len(self._char_idx2vocab) >= self.charsNum:
                self._char_idx2vocab = self._char_idx2vocab[: self.charsNum]
            else:
                self.charsNum = len( self._char_idx2vocab )
        self._char_vocab2idx = {v:i for i,v in enumerate(self._char_idx2vocab)}
        self.logger.debug( "char_dict size:{}; chars num:{}.".format( len(self._char_vocab2idx), self.charsNum ) )
        return new_wordDict
    # end

    # get the word-count dict or char-count dict.
    def getDict(self, dictName): 
        word_dict = {  }
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + dictName )
        self.logger.debug( "absolute_dataPath: {}".format( absolute_dataPath ) )
        with open( absolute_dataPath, 'r') as dictFile:
            for line in dictFile.readlines():
                line_list = line.rstrip().split()
                # assert len(line_list)!=2
                if len(line_list) !=2 :
                    self.logger.debug( line_list )
                    continue
                word = line_list[0]
                count = int( line_list[1] )
                word_dict[word] = count
        return word_dict
    # end
    
    # 
    def get_sample_num(self):
        # sample_num_dict = self.sample_num
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + "dataSet_describe.txt" )
        with open( absolute_dataPath, 'r' ) as describe_file:
            sample_num = json.loads( describe_file.read() )
        self.sample_num = sample_num
        self.logger.info( "sample_num: {}".format( self.sample_num ) )
        return sample_num
    #

    # train w2v model
    def train_w2v_model(self, file_name_list=['new_train.tsv', 'new_dev.tsv', 'new_test.tsv'] ):
        res_file_name = ""
        for file_name in file_name_list:
            res_file_name += file_name.split('.')[0] + "__"

        dataPath = self.dataPath
        sentences = SentenceIters( dataPath=dataPath, file_name_list=file_name_list )
        min_count = 5
        size_dim = 300
        iter_times = 5
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.logger.debug( "-------- {} -------".format( "train word2vec model !!! " ) )
        model = Word2Vec( sentences, window=5, min_count=min_count, size=size_dim, workers=4, iter=iter_times )
        model_name = "{}mn_count{}_{}d_{}iter".format( res_file_name, min_count, size_dim, iter_times )
        # model.save( os.path.join( ProjectRootDIR, dataPath + model_name ) )
        model.save( os.path.join( ProjectRootDIR, dataPath + model_name +'.model' ) )
        model.wv.save_word2vec_format( os.path.join( ProjectRootDIR, dataPath + model_name +'.txt' ), binary=False)
    # end 
    
    def load_w2v(self, w2v_name='new_train__new_dev__new_test__mn_count5_300d_5iter.txt'):
        absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + w2v_name )
        wv_from_text = KeyedVectors.load_word2vec_format( absolute_dataPath )
        w2v_dim = wv_from_text.vector_size
        word_vec_matrix = np.random.normal( 0, 0.001, w2v_dim*len(self._word_vocab2idx) )\
                                   .reshape( len(self._word_vocab2idx), w2v_dim )
        for word,idx in self._word_vocab2idx.items():
            if word not in wv_from_text.wv.vocab:
                continue
            word_vec_matrix[ idx:idx+1 ] = wv_from_text.wv[ word ]
        return word_vec_matrix
    # end: return word_vec_matrix

    # 逐行读取数据，并做初步的筛选 特殊字符；   可另加其他操作，如：分词，去停用词，词性替换，同义词替换，等等；
    # read the data one line by one line; and filter the stop char;  adding another handler, like: participate word, filter stop-words...
    def prepCsvData( self, file_name_list=['train.tsv', 'dev.tsv', 'test.tsv'],  
                           s1_key='text1', s2_key='text2', label_key='label' ):
        for file_name in file_name_list:
            absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + "new_"+file_name )
            if os.path.exists( absolute_dataPath ):
                self.logger.debug( "The file: {} exits.".format( self.dataPath + "new_"+file_name ) )
                continue
            with open( absolute_dataPath, 'w' ) as writerFile:
                fieldnames = [ 'q1', 'q2', 'label' ]
                writer = csv.DictWriter( writerFile, delimiter='\t', fieldnames=fieldnames )
                writer.writeheader()
                absolute_dataPath = os.path.join( ProjectRootDIR, self.dataPath + file_name)
                with open( absolute_dataPath, 'r' ) as csvfile:
                    csvReader = csv.DictReader(csvfile, delimiter='\t')
                    for row in csvReader:
                        q1 = self.__handler( row[ s1_key ]  )
                        q2 = self.__handler( row[ s2_key ] )
                        label = row[ label_key ]
                        writer.writerow({'q1': q1, 'q2': q2, 'label': label })
    # end

    # __handler for ques_str ( per line ); like: filter the stop word, replace the special char;
    def __handler(self, ques_str):
        ques_str = str(ques_str).lower()
        stop_char_list = [ '"', '/', "'", '(', ')' ]
        replace_char = [ ',', '.', '?' ]
        for char in replace_char:
            ques_str = ques_str.replace( char, ' {} '.format( char ) ).replace( '  ', ' ' ).replace( '  ', ' ' )
        for char in stop_char_list:
            ques_str = ques_str.replace( char, '' )
        return ques_str
    # end

# class end


if __name__=='__main__':
    
    prep_data = PrepData(  dataPath=str('dataSets/quora/')  )
    prep_data.prepCsvData() 
    prep_data.save_Word_Char_Dict( file_name_list=['new_train.tsv', 'new_dev.tsv', 'new_test.tsv'] )
    word_dict = prep_data.getDict(  dictName='word_count_dict.txt'  )
    char_dict = prep_data.getDict(  dictName='char_count_dict.txt'  )
    prep_data.filterWordCharDict_saveVoc2idx(  word_dict, char_dict  )
    prep_data.train_w2v_model()
    word_embeddings = prep_data.load_w2v()
    

    prep_data.get_sample_num()
    print( prep_data.sample_num )

    # data_iters = prep_data.get_BatchData( file_name='new_train.tsv', batch_size=512, with_char_inputs=True  )
    # index = 0 
    # for batch_data in data_iters:
    #     for key,val in batch_data.items():
    #         if not isinstance( val, bool ):
    #             print( "{}: {}".format( key, len(val)  ) )
    #     print( "{}--{}".format( index, len( batch_data['s1_words_inputs'])  ) )
    #     time.sleep( 5 )
    #     index +=1 
    # print( "batch num:{}".format( index ) )
    

