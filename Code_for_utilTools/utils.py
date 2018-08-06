# coding:utf-8
import yaml

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







