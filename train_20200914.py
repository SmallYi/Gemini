import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils_20200914 import *
import os
import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='1',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=5,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=5,
        help='batch size')
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--result_folder', type=str,
        default='./result', help='path for test result saving')
parser.add_argument('--log_path', type=str, default='console.txt',
        help='path for training log')




if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    RESULT_FOLDER = args.result_folder
    LOG_PATH = args.log_path



    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    DATA_FILE_NAME = './data/virus/'
    SOFTWARE=('openssl-',)
    OPTIMIZATION=('-o0', '-o1','-o2','-o3')
    COMPILER=('arm-linux', 'x86-linux', 'mips-linux')
    VERSION=('v54',)

    
    NotInclude = ['openssl101ggcc540.json','openssl101gclang380.json']

    test_folder = NotInclude[0].split('.json')[0]+'_'+NotInclude[1].split('.json')[0]
    RESULT_FOLDER = os.path.abspath(os.path.join(RESULT_FOLDER, test_folder))
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    LOG_PATH = os.path.abspath(os.path.join(RESULT_FOLDER, LOG_PATH))


    VALID_F_NAME = []
    valid_filename_str = ""
    for f in NotInclude:
        file_path = DATA_FILE_NAME+f
        VALID_F_NAME.append(file_path)
        valid_filename_str =  valid_filename_str + file_path + '#'



    FUNC_NAME_DICT = {}
    VALID_FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
            OPTIMIZATION, VERSION, NotInclude)

    print 'valid_filename:',valid_filename_str

    FUNC_NAME_DICT = get_f_dict(F_NAME)
    VALID_FUNC_NAME_DICT = get_f_dict(VALID_F_NAME)

    # Binary_Funcname_Dict = get_binary_f_dict(F_NAME)
    # for item in Binary_Funcname_Dict.keys():
    #     print item,Binary_Funcname_Dict[item]

    Gs_train, classes_train = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    Gs_dev, classes_dev = read_graph(VALID_F_NAME, VALID_FUNC_NAME_DICT, NODE_FEATURE_DIM)


    print "Train: {} graphs, {} functions".format(
            len(Gs_train), len(classes_train))
    print "Dev: {} graphs, {} functions".format(
            len(Gs_dev), len(classes_dev))


    # # Fix the pairs for validation
    # if os.path.isfile('data/valid.json'):
    #     with open('data/valid.json') as inf:
    #         valid_ids = json.load(inf)
    #     valid_epoch = generate_epoch_pair(
    #             Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    # else:
    valid_epoch, valid_ids = generate_epoch_pair(
            Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
    with open('data/valid.json', 'w') as outf:
        json.dump(valid_ids, outf)

    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)
    gnn.say("Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train)))
    gnn.say("Dev: {} graphs, {} functions".format(len(Gs_dev), len(classes_dev)))

    gnn.say("Valid_filename:"+valid_filename_str)

    # Train
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, RESULT_FOLDER,load_data=valid_epoch)
    gnn.say("Initial training auc = {0} @ {1}".format(auc, datetime.now()))
    #auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
    #        BATCH_SIZE, load_data=valid_epoch)
    #gnn.say("Initial validation auc = {0} @ {1}".format(auc0, datetime.now()))
    gnn.say("Initial validation auc = {0} @ {1}".format(auc, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("EPOCH {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, RESULT_FOLDER,load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, RESULT_FOLDER,load_data=valid_epoch)
            gnn.say("Testing model: validation auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
