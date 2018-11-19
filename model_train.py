import os
import tensorflow as tf
from model import Model
from hyperparams import params
from data_helper import *

'''
IN GOOGLE COLAB ENV
from google.colab import drive
drive.mount('/content/drive')

import sys
!pip install gensim
!apt-get update
!apt-get install g++ openjdk-8-jdk python-dev python3-dev
!pip3 install JPype1-py3
!pip3 install konlpy
!JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
sys.path.insert(0, '/content/drive/My Drive/Colab Notebooks/')
sys.path.insert(0, '/content/drive/My Drive/Colab Notebooks/Movie Review')
'''

flags = tf.flags
flags.DEFINE_integer('num_units', 128, 'number of LSTM units')
flags.DEFINE_integer('hidden_units', 128, 'number of hidden units in ffn')
flags.DEFINE_integer('num_classes', 2, 'number of classes')
flags.DEFINE_integer('epochs', 4, 'epochs')
flags.DEFINE_integer('batch_size', params['batch_size'], 'batch_size')
flags.DEFINE_float('lr', params['learning_rate'], 'batch_size')
flags.DEFINE_string('dataset', params['default_training_data_path'], 'training data path')
flags.DEFINE_string('testset', params['default_test_data_path'], 'test data path')
config = flags.FLAGS

word_dict = making_word_dictionary()
word_emb = making_word_embedding(word_dict)

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    print('build model...')
    model = Model(config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(config.epochs):
        for i, data in enumerate(datset_iterator(config.dataset, word_dict, config.batch_size)):
            feed_dict={
                model.context : data[0],
                model.lr:config.lr-i/5000,
                model.seq_len:data[1],
                model.labels:data[2]
            }
            _, losses = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        print('training loss = ', losses)
    print('training Done ...')
    print()
    print('Accuracy calculation started ...')

    with open(config.testset, 'r', encoding='utf8') as f:
        total_size = len(f.read().split('\n')[1:-1])
        test_batch = int(total_size / 300) + 1

    total_acc = 0
    for i,data in enumerate(datset_iterator(config.testset, word_dict, config.batch_size)):
        feed_dict = {
            model.context:data[0],
            model.lr:config.lr, #not important
            model.seq_len:data[1],
            model.labels:data[2]    #not important
        }
        acc = sess.run(model.accuracy, feed_dict=feed_dict)
        total_acc += acc/test_batch
    print('test Done... Accuracy : ', total_acc)

    print('Saving started...')
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(params['SAVER_DIR'], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(params['SAVER_DIR'])
    saver.save(sess, checkpoint_path)
    print('Saved well : ', params['SAVER_DIR'])
