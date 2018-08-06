# NLP Sentence Matching   ( tensorflow )

This respository is for the sentence matching question. There are mainly several classic paper method reimplement.

dataSets download: [dataSets.zip](https://drive.google.com/file/d/15FTuNdF5YswRMo3USL_lT_h1hI3ylrAc/view?usp=sharing)



Project Structure:

    |--Code_for_data
        |--prep_data.py
    |--Code_for_model
        |--BIMPM
            |--checkpoints
            |--config
            |--log
            |--src
            |--train_test_BiMPM.py
        |--DecomAtten
            |--checkpoints
            |--config
            |--log
            |--src
            |--train_test_BiMPM.py
        |--
    |--Code_for_utilTools
        |--logger.py
        |--utils.py
    |--dataSets
        |--atec
        |--msr
        |--quora
    |--config_project_rootDir.py
    |--README.md




Code Refer:
1. BiMPM: https://github.com/zhiguowang/BiMPM
2. DecomAtten: https://github.com/shuuki4/decomposable_attention




Paper List:
1. BiMPM: https://arxiv.org/pdf/1702.03814.pdf
2. DecomAtten: https://arxiv.org/pdf/1606.01933.pdf
3. 







### Reference

1. Nikhil Dandekar. https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
2. Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry P. Heck, [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf), **CIKM**, 2013
3. Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, Grégoire Mesnil, [Learning Semantic Representations using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf), **WWW**, 2014
4. Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning, [A Large Annotated Corpus for Learning Natural Language Inference](https://arxiv.org/pdf/1508.05326), **EMNLP**, 2015
5. Aliaksei Severyn, Alessandro Moschitti. [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf), **SIGIR**, 2015
6. Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit: [A decomposable attention model for natural language inference](https://arxiv.org/pdf/1606.01933.pdf) , **EMNLP**, 2016
7. Jiafeng Guo, Yixing Fan, Qingyao Ai, W. Bruce Croft [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/pdf/1711.08611), **CIKM**, 2016
8. Wenpeng Yin, Hinrich Schütze, Bing Xiang, Bowen Zhou, [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf), **TACL**, 2016
9. Zhiguo Wang, Wael Hamza, Radu Florian, [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf), **IJCAI**, 2017
10. Seonhoon Kim, Jin-Hyuk Hong, Inho Kang, Nojun Kwak, [Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information](https://arxiv.org/pdf/1805.11360), CoRR, 2018

