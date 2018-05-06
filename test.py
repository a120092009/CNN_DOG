import tensorflow as tf

import vgg19

import numpy as np
from skimage import io, transform, color

import scipy.io as sio
import xml.etree.ElementTree as ET
import cv2 as cv
import matplotlib.pyplot as plt


def get_bboxs(dirpath, annotation):
    tree = ET.parse(dirpath + annotation)
    root = tree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    lists = []
    for neighbor in root.iter('xmin'):
        xmin.append(neighbor.text)
    for neighbor in root.iter('ymin'):
        ymin.append(neighbor.text)
    for neighbor in root.iter('xmax'):
        xmax.append(neighbor.text)
    for neighbor in root.iter('ymax'):
        ymax.append(neighbor.text)
    lists.append(xmin)
    lists.append(ymin)
    lists.append(xmax)
    lists.append(ymax)
    lists = np.asarray(lists, np.int32)
    return lists


def read_test(annot_path, img_path, cell):
    w = 224
    h = 224
    images = cell[0]
    annotation = cell[1]
    labels = cell[2]
    imgs = []
    label = []
    bboxs = []
    for i in range(100):
        print("Reading test image: " + images[i][0][0])
        img = cv.imread(img_path + images[i][0][0])
        bbox = get_bboxs(annot_path, annotation[i][0][0])

        for idx in range(bbox.shape[1]):
            imgp = img[bbox[1, idx]: bbox[3, idx], bbox[0, idx]: bbox[2, idx]]
            if imgp.shape[0] == 0 or imgp.shape[1] == 0:
                print("ERROR")
                return 0
            imgp = transform.resize(imgp, (w, h), 1, 'constant')
            imgs.append(imgp)
            label.append(labels[i])

    return np.asarray(imgs, np.float32), np.asarray(label, np.int32), len(np.unique(label))


test = sio.loadmat('../data/test_data.mat')

cell_test = test['test_info'][0][0]

data_test, label_test, num_class = read_test('../Annotation/', '../Images/', cell_test)

y_test = np.zeros((np.shape(data_test)[0], 120))
for i in range(np.shape(data_test)[0]):
    y_test[i, label_test[i] - 1] = 1


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, 120])

vgg = vgg19.Vgg19('./test-save best.npy')
vgg.build(images)
with tf.name_scope('loss'):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc8, labels=true_out)
    cost = tf.reduce_mean(cost)

correct_prediction_1 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 1)
correct_prediction_2 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 2)
correct_prediction_3 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 3)
correct_prediction_4 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 4)
correct_prediction_5 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 5)

acc_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))
acc_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
acc_3 = tf.reduce_mean(tf.cast(correct_prediction_3, tf.float32))
acc_4 = tf.reduce_mean(tf.cast(correct_prediction_4, tf.float32))
acc_5 = tf.reduce_mean(tf.cast(correct_prediction_5, tf.float32))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    batch_size = 64

    k = [i for i in range(1, 6)]
    accuracy = []
    test_loss, test_acc_1, test_acc_2, test_acc_3, test_acc_4, test_acc_5, n_batch = 0, 0, 0, 0, 0, 0, 0
    correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), k)
    tf.global_variables_initializer().run()
    for x_test_a, y_test_a in minibatches(data_test, y_test, batch_size, False):
        err, ac1, ac2, ac3, ac4, ac5 = sess.run([cost, acc_1, acc_2, acc_3, acc_4, acc_5],
                                                    feed_dict={images: x_test_a, true_out: y_test_a})

        test_loss = test_loss + err
        test_acc_1 = test_acc_1 + ac1
        test_acc_2 = test_acc_2 + ac2
        test_acc_3 = test_acc_3 + ac3
        test_acc_4 = test_acc_4 + ac4
        test_acc_5 = test_acc_5 + ac5
        n_batch = n_batch + 1
        print(n_batch)
        # print("numbatch: %d, loss: %g,acc: %g" % (n_batch, err, ac))

    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc_5 / n_batch))
    accuracy.append(test_acc_1)
    accuracy.append(test_acc_2)
    accuracy.append(test_acc_3)
    accuracy.append(test_acc_4)
    accuracy.append(test_acc_5)

accuracy = np.asarray(accuracy, np.float32)import tensorflow as tf

import vgg19

import numpy as np
from skimage import io, transform, color

import scipy.io as sio
import xml.etree.ElementTree as ET
import cv2 as cv
import matplotlib.pyplot as plt


def get_bboxs(dirpath, annotation):
    tree = ET.parse(dirpath + annotation)
    root = tree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    lists = []
    for neighbor in root.iter('xmin'):
        xmin.append(neighbor.text)
    for neighbor in root.iter('ymin'):
        ymin.append(neighbor.text)
    for neighbor in root.iter('xmax'):
        xmax.append(neighbor.text)
    for neighbor in root.iter('ymax'):
        ymax.append(neighbor.text)
    lists.append(xmin)
    lists.append(ymin)
    lists.append(xmax)
    lists.append(ymax)
    lists = np.asarray(lists, np.int32)
    return lists


def read_test(annot_path, img_path, cell):
    w = 224
    h = 224
    images = cell[0]
    annotation = cell[1]
    labels = cell[2]
    imgs = []
    label = []
    bboxs = []
    for i in range(100):
        print("Reading test image: " + images[i][0][0])
        img = cv.imread(img_path + images[i][0][0])
        bbox = get_bboxs(annot_path, annotation[i][0][0])

        for idx in range(bbox.shape[1]):
            imgp = img[bbox[1, idx]: bbox[3, idx], bbox[0, idx]: bbox[2, idx]]
            if imgp.shape[0] == 0 or imgp.shape[1] == 0:
                print("ERROR")
                return 0
            imgp = transform.resize(imgp, (w, h), 1, 'constant')
            imgs.append(imgp)
            label.append(labels[i])

    return np.asarray(imgs, np.float32), np.asarray(label, np.int32), len(np.unique(label))


test = sio.loadmat('../data/test_data.mat')

cell_test = test['test_info'][0][0]

data_test, label_test, num_class = read_test('../Annotation/', '../Images/', cell_test)

y_test = np.zeros((np.shape(data_test)[0], 120))
for i in range(np.shape(data_test)[0]):
    y_test[i, label_test[i] - 1] = 1


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, 120])

vgg = vgg19.Vgg19('./test-save best.npy')
vgg.build(images)
with tf.name_scope('loss'):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc8, labels=true_out)
    cost = tf.reduce_mean(cost)

correct_prediction_1 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 1)
correct_prediction_2 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 2)
correct_prediction_3 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 3)
correct_prediction_4 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 4)
correct_prediction_5 = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 5)

acc_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))
acc_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
acc_3 = tf.reduce_mean(tf.cast(correct_prediction_3, tf.float32))
acc_4 = tf.reduce_mean(tf.cast(correct_prediction_4, tf.float32))
acc_5 = tf.reduce_mean(tf.cast(correct_prediction_5, tf.float32))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    batch_size = 64

    k = [i for i in range(1, 6)]
    accuracy = []
    test_loss, test_acc_1, test_acc_2, test_acc_3, test_acc_4, test_acc_5, n_batch = 0, 0, 0, 0, 0, 0, 0
    correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), k)
    tf.global_variables_initializer().run()
    for x_test_a, y_test_a in minibatches(data_test, y_test, batch_size, False):
        err, ac1, ac2, ac3, ac4, ac5 = sess.run([cost, acc_1, acc_2, acc_3, acc_4, acc_5],
                                                    feed_dict={images: x_test_a, true_out: y_test_a})

        test_loss = test_loss + err
        test_acc_1 = test_acc_1 + ac1
        test_acc_2 = test_acc_2 + ac2
        test_acc_3 = test_acc_3 + ac3
        test_acc_4 = test_acc_4 + ac4
        test_acc_5 = test_acc_5 + ac5
        n_batch = n_batch + 1
        print(n_batch)
        # print("numbatch: %d, loss: %g,acc: %g" % (n_batch, err, ac))

    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc_5 / n_batch))
    accuracy.append(test_acc_1)
    accuracy.append(test_acc_2)
    accuracy.append(test_acc_3)
    accuracy.append(test_acc_4)
    accuracy.append(test_acc_5)

accuracy = np.asarray(accuracy, np.float32)
print(accuracy.shape)
plt.figure()
plt.plot(k, accuracy / n_batch)
plt.xlabel('Rank')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Rank')
plt.show()
print(accuracy.shape)
plt.figure()
plt.plot(k, accuracy / n_batch)
plt.xlabel('Rank')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Rank')
plt.show()