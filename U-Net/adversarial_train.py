# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import argparse
import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from tensorflow.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet
from attacks import FastGradientMethod, ProjectedGradientDescent, MomentumIterativeMethod, GaussianNoise

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

def produce_adv(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_adv   = config['adversarial']
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']

    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)

    # 2, construct graph
    print("Constructing graph...")
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64,   shape = full_label_shape)

    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby    = tf.nn.softmax(predicty)

    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    print("old y = %s" % str(y))
    print("old logits = %s" % str(predicty))
    print('size of predicty:', predicty)


    # 3, initialize session and saver
    with tf.Session() as sess:
        saver = tf.train.Saver()
        adv_method = config_adv['method']
        if adv_method == 'FastGradientMethod':
            attack = FastGradientMethod(net)
        elif adv_method == 'MomentumIterativeMethod':
            attack = MomentumIterativeMethod(net)
        elif adv_method == 'ProjectedGradientDescent':
            attack = ProjectedGradientDescent(net)
        elif adv_method == 'GaussianNoise':
            attack = GaussianNoise(net)
        else:
            print("Unknown adversary %s" % adv_method)
            exit()
        #fgsm_params = {'eps': 0.01}
        #adv_x = fgsm.generate(x, **fgsm_params)

        adv_steps = config_adv['iterations']  if adv_method == 'FastGradientMethod' else 1
        adv_eps = config_adv['eps']

        if config_adv['targeted']:
            # Targeted attack to make NN label as no tumor
            params = {'eps': adv_eps/adv_steps, 'y_target': tf.zeros_like(y), 'loss_func': lambda logits, labels: loss_func(logits, labels, weight_map = w)}
        else:
            params = {'eps': adv_eps/adv_steps, 'y': y, 'loss_func': lambda logits, labels: loss_func(logits, labels, weight_map = w)}

        if adv_method != 'FastGradientMethod':
            params['nb_iter'] = config_adv['iterations']

        print("Generating %s attack with method %s; eps=%f iterations=%d" % ("targeted" if config_adv['targeted'] else "", adv_method, adv_eps, config_adv['iterations']))
        print("Params: %s" % str(params))
        adv_x = attack.generate(x, **params)

        sess.run(tf.global_variables_initializer())


        dataloader = DataLoader(config_data)
        dataloader.load_data()

        # 4, start to train
        print("Restoring model...")
        loss_file = config_train['model_save_prefix'] + "_loss.txt"
        saver.restore(sess, config_train['model_pre_trained'])

        num_images = dataloader.get_total_image_number()
        print("Running adversary on %d images" % num_images)
        # calculated dice scores
        batch_dice_list = []
        batch_dice_list_adv = []
        for n in range(num_images):
            print("FGSM img " + str(n + 1))
            #[temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
            train_pair = dataloader.get_subimage_batch()
            tempx = train_pair['images']
            adv = np.copy(tempx)
            tempw = train_pair['weights']
            tempy = train_pair['labels']

            for i in range(adv_steps):
                with sess.as_default():
                    adv = sess.run(adv_x, feed_dict={x:adv, y:tempy, w:tempw})

            print("Saving inputs and outputs...")

            for i in range(batch_size):
                save_array_as_nifty_volume(tempx[i], config_adv['save_folder']+"/img_{0:}.nii.gz".format(n*batch_size + i))
                save_array_as_nifty_volume(adv[i], config_adv['save_folder']+"/img_{0:}_adv.nii.gz".format(n*batch_size + i))

            label_og = sess.run(proby, feed_dict={x:tempx, w:tempw})
            label_adv = sess.run(proby, feed_dict={x:adv, w:tempw})

            save_label = np.asarray(tempy, np.float32)
            for i in range(batch_size):
                save_array_as_nifty_volume(label_og[i], config_adv['save_folder']+"/label_{0:}.nii.gz".format(n*batch_size + i))
                save_array_as_nifty_volume(label_adv[i], config_adv['save_folder']+"/label_{0:}_adv.nii.gz".format(n*batch_size + i))
                save_array_as_nifty_volume(save_label[i], config_adv['save_folder']+"/label_{0:}_true.nii.gz".format(n*batch_size + i))

            # Calculate dice scores
            loss_tempx = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
            loss_adv = loss.eval(feed_dict ={x:adv, w:tempw, y:tempy})
            print("OG loss: %f Adv loss: %f" % (loss_tempx, loss_adv))
            batch_dice_list.append(loss_tempx)
            batch_dice_list_adv.append(loss_adv)

        batch_dice_og = np.asarray(batch_dice_list, np.float32).mean()
        batch_dice_adv = np.asarray(batch_dice_list_adv, np.float32).mean()
        t = time.strftime('%X %x %Z')
        print(t, 'n', n,'loss_og', batch_dice_og, 'loss_adv', batch_dice_adv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="test cases config file. example: python train.py kpa1config/test_wt_ax_adv.txt")
    args = parser.parse_args()
    assert(os.path.isfile(args.config_file))
    produce_adv(args.config_file)
