#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import itertools
import pickle
import time
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from collections import namedtuple, OrderedDict
from tqdm import tqdm

from tensorflow.python import keras
from tensorflow.python.keras.layers import Activation, Add, AveragePooling2D, Conv2D, Dense, BatchNormalization, Flatten, MaxPooling2D, Lambda
import scipy.integrate as it
from absl import flags

# Adapted from https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py
GATE_OP = 1


class PCGrad(tf.compat.v1.train.Optimizer):
    """PCGrad. https://arxiv.org/pdf/2001.06782.pdf."""

    def __init__(self, opt, use_locking=False, name="PCGrad"):
        """optimizer: the optimizer being wrapped."""
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = opt

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        assert isinstance(loss, list)
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat(
            [tf.reshape(grad, [-1, ]) for grad in tf.gradients(
                x, var_list) if grad is not None], axis=0), loss)

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(
                    grads_task[k] * grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod(
                    [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(flags.FLAGS)

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('steps', 100, 'Number of epoch to train.')
tf.compat.v1.flags.DEFINE_integer('batch_size', 256, 'Number of examples in a minibatch.')
tf.compat.v1.flags.DEFINE_integer('order', -1, 'Order of permutations to consider.')
tf.compat.v1.flags.DEFINE_enum('eval', 'test', ['valid', 'test'], 'The eval dataset.')
tf.compat.v1.flags.DEFINE_enum('method', 'mtl', ['mtl', 'fast_mtl'], 'Multitask Training Method.')
tf.compat.v1.flags.DEFINE_list('tasks', ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                            'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                            'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                            'Wearing_Necktie', 'Young'], "The attributes to predict in CelebA.")
SEED = 0
METRICS_AVERAGE = 1
EPSILON = 0.001
TRAIN_SIZE = 162770
VALID_SIZE = 19867
TEST_SIZE = 19962


class ResBlock(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, strides, name):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(
            filters=filters[0],
            kernel_size=kernel_size[0],
            strides=strides,
            name='conv{}_1'.format(name),
            kernel_initializer=keras.initializers.glorot_uniform(seed=SEED),
            padding='same',
            use_bias=False)
        self.bn1 = BatchNormalization(axis=3, name='bn{}_1'.format(name))
        self.conv2 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size[1],
            strides=(1, 1),
            name='conv{}_2'.format(name),
            kernel_initializer=keras.initializers.glorot_uniform(seed=SEED),
            padding='same',
            use_bias=False)
        self.bn2 = BatchNormalization(axis=3, name='bn{}_2'.format(name))

        if strides == (1, 1):
            self.shortcut = Lambda(lambda x: x)
        else:
            self.shortcut = tf.keras.Sequential()
            shortcut_conv = Conv2D(filters=filters[1],
                                   kernel_size=1,
                                   strides=(2, 2),
                                   name='skip_conv{}_1'.format(name),
                                   kernel_initializer=keras.initializers.glorot_uniform(seed=SEED),
                                   padding='valid',
                                   use_bias=False)
            shortcut_bn = BatchNormalization(axis=3, name='skip_bn{}_1'.format(name))
            self.shortcut.add(shortcut_conv)
            self.shortcut.add(shortcut_bn)

    def call(self, inputs):
        x = inputs
        x = Activation('relu')(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = Add()([x, self.shortcut(inputs)])
        return Activation('relu')(x)


class ResNet18(tf.keras.Model):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1_1 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            name='conv1_1',
            kernel_initializer=keras.initializers.glorot_uniform(seed=SEED),
            padding='same',
            use_bias=False)
        self.bn1_1 = BatchNormalization(axis=3, name='bn1_1')
        self.resblock_2 = ResBlock([64, 64], [3, 3], (1, 1), '1')

    def call(self, inputs):
        x = inputs
        x = Activation('relu')(self.bn1_1(self.conv1_1(x)))
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = self.resblock_2(x)
        x = AveragePooling2D((2, 2), name='avg_pool')(x)
        x = Flatten()(x)
        return x


class AttributeDecoder(tf.keras.Model):#this is the task specific parameters
    def __init__(self):
        super(AttributeDecoder, self).__init__()
        self.fc1 = Dense(2, kernel_initializer=keras.initializers.glorot_uniform(seed=SEED))

    def call(self, inputs):
        x = inputs
        x = self.fc1(x)
        return x


def res_block_step(inputs, base_updated):
    conv1 = tf.nn.conv2d(inputs, base_updated[0], strides=(2, 2), padding="SAME")
    mean1, variance1 = tf.nn.moments(conv1, axes=[0, 1, 2])
    gamma1, beta1 = base_updated[1], base_updated[2]
    bn_conv1 = tf.nn.batch_normalization(conv1, mean1, variance1, offset=beta1, scale=gamma1, variance_epsilon=EPSILON)
    relu1 = tf.nn.relu(bn_conv1)

    conv2 = tf.nn.conv2d(relu1, base_updated[3], strides=(1, 1), padding="SAME")
    mean2, variance2 = tf.nn.moments(conv2, axes=[0, 1, 2])
    gamma2, beta2 = base_updated[4], base_updated[5]
    bn_conv2 = tf.nn.batch_normalization(conv2, mean2, variance2, offset=beta2, scale=gamma2, variance_epsilon=EPSILON)

    skip_conv = tf.nn.conv2d(inputs, base_updated[6], strides=(2, 2), padding="VALID")
    skip_mean, skip_variance = tf.nn.moments(skip_conv, axes=[0, 1, 2])
    skip_gamma, skip_beta = base_updated[7], base_updated[8]
    skip_bn = tf.nn.batch_normalization(skip_conv, skip_mean, skip_variance, offset=skip_beta, scale=skip_gamma,
                                        variance_epsilon=EPSILON)

    res_block = tf.nn.relu(bn_conv2 + skip_bn)
    return res_block


def base_step(inputs, base_updated):
    # ResNet Block 1 Output.
    conv1_1 = tf.nn.conv2d(inputs, base_updated[0], strides=(1, 1), padding="SAME")
    mean1_1, variance1_1 = tf.nn.moments(conv1_1, axes=[0, 1, 2],
                                         keepdims=True)  # normalize across the channel dimension for spacial batch norm.
    gamma1_1, beta1_1 = base_updated[1], base_updated[2]
    bn_conv1_1 = tf.nn.batch_normalization(conv1_1, mean1_1, variance1_1, offset=beta1_1, scale=gamma1_1,
                                           variance_epsilon=EPSILON)
    res_block_1 = tf.nn.max_pool2d(tf.nn.relu(bn_conv1_1), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # ResNet Block 2
    conv2_1 = tf.nn.conv2d(res_block_1, base_updated[3], strides=(1, 1), padding="SAME")
    mean2_1, variance2_1 = tf.nn.moments(conv2_1, axes=[0, 1, 2])
    gamma2_1, beta2_1 = base_updated[4], base_updated[5]
    bn_conv2_1 = tf.nn.batch_normalization(conv2_1, mean2_1, variance2_1, offset=beta2_1, scale=gamma2_1,
                                           variance_epsilon=EPSILON)
    res_block2_1 = tf.nn.relu(bn_conv2_1)

    conv2_2 = tf.nn.conv2d(res_block2_1, base_updated[6], strides=(1, 1), padding="SAME")
    mean2_2, variance2_2 = tf.nn.moments(conv2_2, axes=[0, 1, 2])
    gamma2_2, beta2_2 = base_updated[7], base_updated[8]
    bn_conv2_2 = tf.nn.batch_normalization(conv2_2, mean2_2, variance2_2, offset=beta2_2, scale=gamma2_2,
                                           variance_epsilon=EPSILON)
    res_block_2 = tf.nn.relu(bn_conv2_2 + res_block_1)

    avg_pool = tf.nn.avg_pool2d(res_block_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    resnet_out = tf.reshape(avg_pool, [inputs.shape[0], -1])
    return resnet_out


def permute(losses):##losses is a directory [task: loss]
    """Returns all combinations of losses in the loss dictionary."""
    losses = OrderedDict(sorted(losses.items()))
    rtn = {}
    for task, loss in losses.items():
        tmp_dict = {task: loss}
        for saved_task, saved_loss in rtn.items():
            if FLAGS.order == 1:
                continue  # Skip higher than first-order combinations.
            new_task = "{}_{}".format(saved_task, task)
            new_loss = loss + saved_loss
            tmp_dict[new_task] = new_loss
        rtn.update(tmp_dict)

    if FLAGS.order == 1:
        rtn["_".join(losses.keys())] = sum(losses.values())##loss of combined nine tasks
    return rtn## individule tasks losses and conbined tasks loss


def permute_list(lst):
    """Returns all combinations of tasks in the task list."""
    lst.sort()
    rtn = []
    for task in lst:
        tmp_lst = [task]
        for saved_task in rtn:
            if FLAGS.order == 1:
                continue
            new_task = "{}_{}".format(saved_task, task)
            tmp_lst.append(new_task)
        rtn += tmp_lst

    if FLAGS.order == 1:
        rtn.append("_".join(lst))
    return rtn


def decay_lr(step, optimizer):
    if (step + 1) % 15 == 0:
        optimizer.lr = optimizer.lr / 2.
        print('Decreasing the learning rate by 1/2. New Learning Rate: {}'.format(optimizer.lr))


def decay_pcgrad_lr(step, lr_var):
    if (step + 1) % 15 == 0:
        lr_var.assign(lr_var / 2.)
        print('Decreasing the learning rate by 1/2.')


def add_average(lst, metrics_dict, n):
    if len(lst) < n:
        lst.append(metrics_dict)
    elif len(lst) == n:
        lst.pop(0)
        lst.append(metrics_dict)
    elif len(lst) > n:
        raise Exception('List size is greater than n. This should never happen.')


def compute_average(metrics_list, n):
    if not metrics_list:
        return {}
    rtn = {task: 0. for task in metrics_list[0]}
    for metric in metrics_list:
        for task in metric:
            rtn[task] += metric[task] / float(n)
    return rtn


def load_dataset(batch_size, data_dir):
    train = tfds.load(name= 'celeb_a',  split='train', download=False, data_dir=data_dir)
    resized_train = train.map(
        lambda d: (d['attributes'], tf.image.resize(tf.image.convert_image_dtype(d['image'], tf.float32), [64, 64])))
    final_train = resized_train.shuffle(
        buffer_size=TRAIN_SIZE, seed=SEED,
        reshuffle_each_iteration=True).batch(batch_size)

    valid = tfds.load(name= 'celeb_a', split='validation',download= False, data_dir=data_dir)
    resized_valid = valid.map(
        lambda d: (d['attributes'], tf.image.resize(tf.image.convert_image_dtype(d['image'], tf.float32), [64, 64])))
    final_valid = resized_valid.batch(batch_size)

    test = tfds.load(name= 'celeb_a', split='test',download= False, data_dir=data_dir)
    resized_test = test.map(
        lambda d: (d['attributes'], tf.image.resize(tf.image.convert_image_dtype(d['image'], tf.float32), [64, 64])))
    final_test = resized_test.batch(batch_size)

    Dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
    return Dataset(final_train, final_valid, final_test)


def get_uncertainty_weights():
    uncertainty_weights = {}
    global shadow_uncertainty
    if shadow_uncertainty is None:
        shadow_uncertainty = tf.Variable(1.0)
    uncertainty_weights['5_o_Clock_Shadow'] = shadow_uncertainty
    global black_hair_uncertainty
    if black_hair_uncertainty is None:
        black_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Black_Hair'] = black_hair_uncertainty
    global blond_hair_uncertainty
    if blond_hair_uncertainty is None:
        blond_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Blond_Hair'] = blond_hair_uncertainty
    global brown_hair_uncertainty
    if brown_hair_uncertainty is None:
        brown_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Brown_Hair'] = brown_hair_uncertainty
    global goatee_uncertainty
    if goatee_uncertainty is None:
        goatee_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Goatee'] = goatee_uncertainty
    global mustache_uncertainty
    if mustache_uncertainty is None:
        mustache_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Mustache'] = mustache_uncertainty
    global no_beard_uncertainty
    if no_beard_uncertainty is None:
        no_beard_uncertainty = tf.Variable(1.0)
    uncertainty_weights['No_Beard'] = no_beard_uncertainty
    global rosy_cheeks_uncertainty
    if rosy_cheeks_uncertainty is None:
        rosy_cheeks_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Rosy_Cheeks'] = rosy_cheeks_uncertainty
    global wearing_hat_uncertainty
    if wearing_hat_uncertainty is None:
        wearing_hat_uncertainty = tf.Variable(1.0)
    uncertainty_weights['Wearing_Hat'] = wearing_hat_uncertainty
    return uncertainty_weights


def init_uncertainty_weights():
    global shadow_uncertainty
    shadow_uncertainty = None
    global black_hair_uncertainty
    black_hair_uncertainty = None
    global blond_hair_uncertainty
    blond_hair_uncertainty = None
    global brown_hair_uncertainty
    brown_hair_uncertainty = None
    global goatee_uncertainty
    goatee_uncertainty = None
    global mustache_uncertainty
    mustache_uncertainty = None
    global no_beard_uncertainty
    no_beard_uncertainty = None
    global rosy_cheeks_uncertainty
    rosy_cheeks_uncertainty = None
    global wearing_hat_uncertainty
    wearing_hat_uncertainty = None


def init_gradnorm_weights():
    global shadow_gradnorm
    shadow_gradnorm = None
    global black_hair_gradnorm
    black_hair_gradnorm = None
    global blond_hair_gradnorm
    blond_hair_gradnorm = None
    global brown_hair_gradnorm
    brown_hair_gradnorm = None
    global goatee_gradnorm
    goatee_gradnorm = None
    global mustache_gradnorm
    mustache_gradnorm = None
    global no_beard_gradnorm
    no_beard_gradnorm = None
    global rosy_cheeks_gradnorm
    rosy_cheeks_gradnorm = None
    global wearing_hat_gradnorm
    wearing_hat_gradnorm = None


def init_gradnorm_weights():
    global shadow_gradnorm
    shadow_gradnorm = None
    global black_hair_gradnorm
    black_hair_gradnorm = None
    global blond_hair_gradnorm
    blond_hair_gradnorm = None
    global brown_hair_gradnorm
    brown_hair_gradnorm = None
    global goatee_gradnorm
    goatee_gradnorm = None
    global mustache_gradnorm
    mustache_gradnorm = None
    global no_beard_gradnorm
    no_beard_gradnorm = None
    global rosy_cheeks_gradnorm
    rosy_cheeks_gradnorm = None
    global wearing_hat_gradnorm
    wearing_hat_gradnorm = None


def fetch_gradnorm_weights():
    gradnorm_weights = {}
    global shadow_gradnorm
    if shadow_gradnorm is None:
        shadow_gradnorm = tf.Variable(1.0)
    gradnorm_weights['5_o_Clock_Shadow'] = shadow_gradnorm
    global black_hair_gradnorm
    if black_hair_gradnorm is None:
        black_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Black_Hair'] = black_hair_gradnorm
    global blond_hair_gradnorm
    if blond_hair_gradnorm is None:
        blond_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Blond_Hair'] = blond_hair_gradnorm
    global brown_hair_gradnorm
    if brown_hair_gradnorm is None:
        brown_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Brown_Hair'] = brown_hair_gradnorm
    global goatee_gradnorm
    if goatee_gradnorm is None:
        goatee_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Goatee'] = goatee_gradnorm
    global mustache_gradnorm
    if mustache_gradnorm is None:
        mustache_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Mustache'] = mustache_gradnorm
    global no_beard_gradnorm
    if no_beard_gradnorm is None:
        no_beard_gradnorm = tf.Variable(1.0)
    gradnorm_weights['No_Beard'] = no_beard_gradnorm
    global rosy_cheeks_gradnorm
    if rosy_cheeks_gradnorm is None:
        rosy_cheeks_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Rosy_Cheeks'] = rosy_cheeks_gradnorm
    global wearing_hat_gradnorm
    if wearing_hat_gradnorm is None:
        wearing_hat_gradnorm = tf.Variable(1.0)
    gradnorm_weights['Wearing_Hat'] = wearing_hat_gradnorm
    return gradnorm_weights


def init_gradnorm_l0():
    global shadow_loss
    shadow_loss = None
    global black_hair_loss
    black_hair_loss = None
    global blond_hair_loss
    blond_hair_loss = None
    global brown_hair_loss
    brown_hair_loss = None
    global goatee_loss
    goatee_loss = None
    global mustache_loss
    mustache_loss = None
    global no_beard_loss
    no_beard_loss = None
    global rosy_cheeks_loss
    rosy_cheeks_loss = None
    global wearing_hat_loss
    wearing_hat_loss = None


def fetch_gradnorm_l0(losses):
    gradnorm_l0 = {}
    global shadow_loss
    if shadow_loss is None:
        if '5_o_Clock_Shadow' in losses:
            loss_val = losses['5_o_Clock_Shadow']
        else:
            loss_val = 0.
        shadow_loss = tf.Variable(loss_val)
    gradnorm_l0['5_o_Clock_Shadow'] = shadow_loss
    global black_hair_loss
    if black_hair_loss is None:
        if 'Black_Hair' in losses:
            loss_val = losses['Black_Hair']
        else:
            loss_val = 0.
        black_hair_loss = tf.Variable(loss_val)
    gradnorm_l0['Black_Hair'] = black_hair_loss
    global blond_hair_loss
    if blond_hair_loss is None:
        if 'Blond_Hair' in losses:
            loss_val = losses['Blond_Hair']
        else:
            loss_val = 0.
        blond_hair_loss = tf.Variable(loss_val)
    gradnorm_l0['Blond_Hair'] = blond_hair_loss
    global brown_hair_loss
    if brown_hair_loss is None:
        if 'Brown_Hair' in losses:
            loss_val = losses['Brown_Hair']
        else:
            loss_val = 0.
        brown_hair_loss = tf.Variable(loss_val)
    gradnorm_l0['Brown_Hair'] = brown_hair_loss
    global goatee_loss
    if goatee_loss is None:
        if 'Goatee' in losses:
            loss_val = losses['Goatee']
        else:
            loss_val = 0.
        goatee_loss = tf.Variable(loss_val)
    gradnorm_l0['Goatee'] = goatee_loss
    global mustache_loss
    if mustache_loss is None:
        if 'Mustache' in losses:
            loss_val = losses['Mustache']
        else:
            loss_val = 0.
        mustache_loss = tf.Variable(loss_val)
    gradnorm_l0['Mustache'] = mustache_loss
    global no_beard_loss
    if no_beard_loss is None:
        if 'No_Beard' in losses:
            loss_val = losses['No_Beard']
        else:
            loss_val = 0.
        no_beard_loss = tf.Variable(loss_val)
    gradnorm_l0['No_Beard'] = no_beard_loss
    global rosy_cheeks_loss
    if rosy_cheeks_loss is None:
        if 'Rosy_Cheeks' in losses:
            loss_val = losses['Rosy_Cheeks']
        else:
            loss_val = 0.
        rosy_cheeks_loss = tf.Variable(loss_val)
    gradnorm_l0['Rosy_Cheeks'] = rosy_cheeks_loss
    global wearing_hat_loss
    if wearing_hat_loss is None:
        if 'Wearing_Hat' in losses:
            loss_val = losses['Wearing_Hat']
        else:
            loss_val = 0.
        wearing_hat_loss = tf.Variable(loss_val)
    gradnorm_l0['Wearing_Hat'] = wearing_hat_loss
    return gradnorm_l0


def compute_gradnorm_losses(losses, gradnorm_l0, gradnorms, expected_gradnorm):
    task_li = {}
    for task in FLAGS.tasks:
        task_li[task] = losses[task] / gradnorm_l0[task]
    li_expected = tf.reduce_mean(list(task_li.values()))

    gradnorm_loss = {}
    for task in FLAGS.tasks:
        task_ri = tf.math.pow(task_li[task] / li_expected, params.alpha)
        gradnorm_loss[task] = tf.norm(gradnorms[task] - tf.stop_gradient(expected_gradnorm * task_ri), ord=1)
    total_gradnorm_loss = tf.reduce_sum(list(gradnorm_loss.values()))
    return total_gradnorm_loss


def train(params):
    print(params)

    ResBase = ResNet18()
    ResTowers = {task: AttributeDecoder() for task in FLAGS.tasks}

    dataset = load_dataset(FLAGS.batch_size, params.data_dir)
    print('Dataset loaded successfully!')
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.keras.optimizers.SGD(params.lr, momentum=0.9)
    if 'pcgrad' in FLAGS.method:
        lr_var = tf.Variable(params.lr)
        old_optimizer = tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9)
        optimizer = PCGrad(tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9))

    @tf.function()
    def train_step(input, labels, first_step=False):
        """This is TAG."""
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
            losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=preds[task]))
                for task in labels}##[task : mean of loss]
            loss = tf.add_n(list(losses.values()))#add all tasks loss together

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            task_gains = {}
            task_permutations = permute(losses)
            combined_task_gradients = [
                (combined_task, tape.gradient(task_permutations[combined_task], ResBase.trainable_weights)) for##task_permutations[combined_task] represents loss value, loss to parameters derivation
                combined_task in task_permutations]

        for combined_task, task_gradient in combined_task_gradients:##combined_task are foucused task
            if first_step:
                base_update = [optimizer.lr * grad for grad in task_gradient]
                base_updated = [param - update for param, update in zip(ResBase.trainable_weights, base_update)]
            else:
                base_update = [(optimizer._momentum * optimizer.get_slot(param, 'momentum') - optimizer.lr * grad)
                               for param, grad in zip(ResBase.trainable_weights, task_gradient)]
                base_updated = [param + update for param, update in zip(ResBase.trainable_weights, base_update)]
            task_update_rep = base_step(input, base_updated)
            task_update_preds = {task: model(task_update_rep, training=True) for (task, model) in ResTowers.items()}
            task_update_losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=task_update_preds[task]))
                for task in labels}
            task_gain = {task: (1.0 - task_update_losses[task] / losses[task]) / optimizer.lr for task in FLAGS.tasks}## why divided by lr
            task_gains[combined_task] = task_gain


        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights) #calculate the weight using sumed up loss, not individual loss
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, task_gains

    @tf.function()
    def cosine_sim_train_step(input, labels, first_step=False):
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
            losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=preds[task]))
                for task in labels}
            loss = tf.add_n(list(losses.values()))

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            task_gains = {}
            task_permutations = permute(losses)

            task_gradients = {}
            for task in FLAGS.tasks:
                task_grads = tape.gradient(task_permutations[task], ResBase.trainable_weights)
                task_grads = [grad for grad in task_grads if len(grad.shape) > 1]
                task_gradients[task] = task_grads
            combined_task_gradients = [
                (combined_task, tape.gradient(task_permutations[combined_task], ResBase.trainable_weights)) for
                combined_task in task_permutations]

        for combined_task, task_gradient in combined_task_gradients:
            task_gain = {}
            for task in FLAGS.tasks:
                filtered_grads = [grad for grad in task_gradient if len(grad.shape) > 1]
                stacked_filtered = tf.concat([tf.reshape(grad, shape=[-1]) for grad in filtered_grads], axis=0)
                stacked_task = tf.concat([tf.reshape(grad, shape=[-1]) for grad in task_gradients[task]], axis=0)
                cosine_sim = tf.reduce_sum(tf.multiply(stacked_filtered, stacked_task)) / (
                            tf.norm(stacked_filtered, ord=2) * tf.norm(stacked_task, ord=2))
                task_gain[task] = cosine_sim
            task_gains[combined_task] = task_gain

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, task_gains

    @tf.function()
    def train_fast_step(input, labels, first_step=False):
        """Call this function to evaluate task groupings. It's faster."""
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
            losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=preds[task]))
                for task in labels}
            loss = tf.add_n(list(losses.values()))

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, {}

    @tf.function()
    def train_uncertainty_step(input, labels, first_step=False):
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
            losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=preds[task]))
                for task in labels}

            uncertainty_weights = get_uncertainty_weights()
            for task in FLAGS.tasks:
                clip_uncertainty = tf.clip_by_value(uncertainty_weights[task], 0.01, 10.0)
                losses[task] = losses[task] / tf.exp(2 * clip_uncertainty) + clip_uncertainty
            loss = tf.add_n(list(losses.values()))

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        # Update the uncertainty weight variables.
        uncertainty_gradients = [val for val in tape.gradient(loss, list(uncertainty_weights.values()))]
        optimizer.apply_gradients(zip(uncertainty_gradients, list(uncertainty_weights.values())))

        global_step.assign_add(1)
        return losses, {}

    @tf.function()
    def train_gradnorm_step(input, labels, first_step=False):
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
            losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels[task],
                logits=preds[task]))
                for task in labels}

            # Essentially gradnorm weights.
            gradnorm_weights = fetch_gradnorm_weights()
            post_weight_losses = {task: gradnorm_weights[task] * losses[task] for task in losses}
            gradnorms = {}
            for task in FLAGS.tasks:
                gradnorms[task] = tf.norm(tape.gradient(post_weight_losses[task], ResBase.trainable_weights[-3]), ord=2)
            expected_gradnorm = tf.add_n(list(gradnorms.values())) / len(FLAGS.tasks)
            loss = tf.add_n(list(post_weight_losses.values()))

            gradnorm_l0 = fetch_gradnorm_l0(losses)
            gradnorm_losses = compute_gradnorm_losses(losses, gradnorm_l0, gradnorms, expected_gradnorm)

        # Update gradnorm weights.
        gradnorm_weight_grads = tape.gradient(gradnorm_losses, gradnorm_weights.values())
        optimizer.apply_gradients(zip(gradnorm_weight_grads, list(gradnorm_weights.values())))

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        # Clip between 0.1 and 10
        for task in FLAGS.tasks:
            gradnorm_weights[task].assign(tf.clip_by_value(gradnorm_weights[task], 0.1, 10.0))

        # Renormalize GradNorm Weights
        scale = len(FLAGS.tasks) / tf.reduce_sum(list(gradnorm_weights.values()))
        for task in FLAGS.tasks:
            gradnorm_weights[task].assign(scale * gradnorm_weights[task])

        global_step.assign_add(1)
        return losses, {}

    @tf.function()
    def train_pcgrad_step(input, labels, first_step=False):
        rep = ResBase(input, training=True)
        preds = {task: model(rep, training=True) for (task, model) in ResTowers.items()}
        losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels[task],
            logits=preds[task]))
            for task in labels}
        uncertainty_weights = get_uncertainty_weights()
        for task in FLAGS.tasks:
            clip_uncertainty = tf.clip_by_value(uncertainty_weights[task], 0.01, 10.0)
            losses[task] = losses[task] / tf.exp(2 * clip_uncertainty) + clip_uncertainty
        loss = tf.add_n(list(losses.values()))

        base_gradvars = optimizer.compute_gradients(list(losses.values()), ResBase.trainable_weights)
        task_gradvars = [optimizer.compute_gradients([losses[task]], model.trainable_weights) for (task, model) in
                         ResTowers.items()]

        old_optimizer.apply_gradients(base_gradvars)
        for gv in task_gradvars:
            old_optimizer.apply_gradients(gv)

        # Update the uncertainty weight variables.
        uw_gv = old_optimizer.compute_gradients(loss, list(uncertainty_weights.values()))
        old_optimizer.apply_gradients(uw_gv)

        global_step.assign_add(1)
        return losses, {}

    @tf.function()
    def eval_step(input, labels):
        rep = ResBase(input)
        preds = {task: ResTowers[task](rep) for (task, model) in ResTowers.items()}
        int_preds = {task: tf.math.argmax(preds[task], 1, tf.dtypes.int32) for task in labels}
        int_labels = {task: tf.math.argmax(labels[task], 1, tf.dtypes.int32) for task in labels}
        losses = {task: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.cast(labels[task], tf.float32),
            logits=preds[task]))
            for task in labels}
        accuracies = {task: tf.math.count_nonzero(tf.equal(int_preds[task], int_labels[task])) for task in labels}
        Eval = namedtuple('Eval', ['losses', 'accuracies'])
        return Eval(losses, accuracies)

    # Training Loop.
    metrics = {'train_loss': [], 'eval_loss': [], 'eval_acc': []}
    gradient_metrics = {task: [] for task in permute_list(FLAGS.tasks)}
    final_metrics = {'train_loss': [], 'eval_loss': [], 'eval_acc': []}
    model_params = []

    end = None
    for step in range(FLAGS.steps):
        if end:
            print(f'Differnece in time: {end - start}')
        start = time.time()
        print('epoch: {}'.format(step))
        if "pcgrad" not in FLAGS.method:
            decay_lr(step, optimizer)  # Halve the learning rate every 30 steps.
        else:
            decay_pcgrad_lr(step, lr_var)
        batch_train_loss = {task: 0. for task in FLAGS.tasks}
        batch_grad_metrics = {combined_task: {task: 0. for task in FLAGS.tasks} for combined_task in gradient_metrics}#combined_task is the foucused task, task is the 9 tasks
        for labels, img in dataset.train:
            labels = {task: tf.keras.utils.to_categorical(labels[task], num_classes=2) for task in labels if
                      task in FLAGS.tasks}
            if FLAGS.method == 'mtl':  # Full TAG.
                losses, task_gains = train_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            elif FLAGS.method == 'cosine_sim_mtl':  # Cosine similarity computation.
                losses, task_gains = cosine_sim_train_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            elif FLAGS.method == 'fast_mtl':  # Runs w/o TAG computation.
                losses, task_gains = train_fast_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            elif FLAGS.method == 'uncertainty_mtl':  # Runs w/ uncertainty weights.
                losses, task_gains = train_uncertainty_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            elif FLAGS.method == 'gradnorm_mtl':  # Runs w/ gradnorm weights.
                losses, task_gains = train_gradnorm_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            elif FLAGS.method == 'pcgrad_mtl':  # Runs w/ PCGrad Optimizer and UW.
                losses, task_gains = train_pcgrad_step(img, labels, first_step=(len(optimizer.variables()) == 0))
            else:
                raise Exception("Unrecognized Method Selected.")

            # Record batch-level training and gradient metrics.
            for combined_task, task_gain_map in task_gains.items():#combined_task is the task foucused, task_gain_map is the assist task and it's affinity
                for task, gain in task_gain_map.items():#task is the assist task and gain is the affinity
                    batch_grad_metrics[combined_task][task] += gain.numpy() / (math.ceil(TRAIN_SIZE / FLAGS.batch_size)) #batch_grad_matrics is a directory like [combined_task : [task: value] ], it is the Z_hat
            for task, loss in losses.items():
                batch_train_loss[task] += loss.numpy() / (math.ceil(TRAIN_SIZE / FLAGS.batch_size))

        # Record epoch-level training and gradient metrics.
        add_average(metrics['train_loss'], batch_train_loss, METRICS_AVERAGE)
        for combined_task, task_gain_map in batch_grad_metrics.items():
            gradient_metrics[combined_task].append(task_gain_map)## this is what we want,contains every epoch task_gain_map. The last one is the final accumulated Z_hat

        batch_eval_loss = {task: 0. for task in FLAGS.tasks}
        batch_eval_acc = {task: 0. for task in FLAGS.tasks}
        for labels, img in dataset.test if FLAGS.eval == 'test' else dataset.valid:
            labels = {task: tf.keras.utils.to_categorical(labels[task], num_classes=2) for task in labels if
                      task in FLAGS.tasks}
            eval_metrics = eval_step(img, labels)
            for task in FLAGS.tasks:
                EVAL_SIZE = TEST_SIZE if FLAGS.eval == 'test' else VALID_SIZE
                batch_eval_loss[task] += eval_metrics.losses[task].numpy() / (math.ceil(EVAL_SIZE / FLAGS.batch_size))
                batch_eval_acc[task] += eval_metrics.accuracies[task].numpy() / EVAL_SIZE
        add_average(metrics['eval_loss'], batch_eval_loss, METRICS_AVERAGE)
        add_average(metrics['eval_acc'], batch_eval_acc, METRICS_AVERAGE)

        for metric in metrics:
            final_metrics[metric].append(compute_average(metrics[metric], METRICS_AVERAGE))

        # Save past EARLY_STOP sets of parameters.
        cur_params = [(
                      'base', copy.deepcopy(ResBase.trainable_weights), copy.deepcopy(ResBase.non_trainable_weights))] + \
                     [(task, copy.deepcopy(tower.trainable_weights), copy.deepcopy(tower.non_trainable_weights)) for
                      task, tower in ResTowers.items()]
        model_params.append(tuple(cur_params))

        # Early stopping. If Validation loss hasn't increased for the past 10 epochs..
        EARLY_STOP = 11
        if step > EARLY_STOP - 1 and all(
                [sum(final_metrics['eval_loss'][-EARLY_STOP].values()) < sum(final_metrics['eval_loss'][-i].values())
                 for i in range(1, EARLY_STOP)]):
            print('Validation loss has not improved for past 10 epochs. Stopping at epoch {}'.format(step))

            # Reload best weights..
            for task, trainables, non_trainables in model_params[-EARLY_STOP]:
                if task == 'base':
                    for param, trainable in zip(ResBase.trainable_weights, trainables):
                        param.assign(trainable)
                    for param, non_trainable in zip(ResBase.non_trainable_weights, non_trainables):
                        param.assign(non_trainable)
                else:
                    for param, trainable in zip(ResTowers[task].trainable_weights, trainables):
                        param.assign(trainable)
                    for param, non_trainable in zip(ResTowers[task].non_trainable_weights, non_trainables):
                        param.assign(non_trainable)

            # Evaluate on the test set.
            batch_test_acc = {task: 0. for task in FLAGS.tasks}
            batch_test_loss = {task: 0. for task in FLAGS.tasks}
            for labels, img in dataset.test:
                labels = {task: tf.keras.utils.to_categorical(labels[task], num_classes=2) for task in labels if
                          task in FLAGS.tasks}
                test_metrics = eval_step(img, labels)
                for task in FLAGS.tasks:
                    EVAL_SIZE = TEST_SIZE if FLAGS.eval == 'test' else VALID_SIZE
                    batch_test_loss[task] += test_metrics.losses[task].numpy() / (
                        math.ceil(EVAL_SIZE / FLAGS.batch_size))
                    batch_test_acc[task] += test_metrics.accuracies[task].numpy() / EVAL_SIZE

            print_test_acc = "\n".join(
                ["{}: {:.2f}".format(task, 100.0 * metric) for task, metric in batch_test_acc.items()])
            print_test_loss = "\n".join(["{}: {:.4f}".format(task, metric) for task, metric in batch_test_loss.items()])
            print("Test Accuracy:\n{}\n".format(print_test_acc))
            print("Test Loss:\n{}\n".format(print_test_loss))

            for task in gradient_metrics:
                gradient_metrics[task] = gradient_metrics[task][:-1 * (EARLY_STOP - 1)]

            return final_metrics, gradient_metrics

        elif len(model_params) == EARLY_STOP:
            model_params.pop(0)

        print_train_loss = "\n".join(
            ["{}: {:.4f}".format(task, metric) for task, metric in final_metrics['train_loss'][-1].items()])
        print("Train Loss:\n{}\n".format(print_train_loss))

        print("grad metrics for fun: {}".format(gradient_metrics))

        print_eval_loss = "\n".join(
            ["{}: {:.4f}".format(task, metric) for task, metric in final_metrics['eval_loss'][-1].items()])
        print("Eval Loss:\n{}\n".format(print_eval_loss))
        print_eval_acc = "\n".join(
            ["{}: {:.2f}".format(task, 100.0 * metric) for task, metric in final_metrics['eval_acc'][-1].items()])
        print("Eval Accuracy:\n{}\n".format(print_eval_acc))
        print("\n-------------\n")
        end = time.time()

    return final_metrics, gradient_metrics

if __name__ == '__main__':
    Params = namedtuple("Params", ['lr', 'alpha', 'data_dir'])  # Params can possibly be tuned, FLAGS can't be tuned.
    params = Params(lr=0.0005, alpha=0.1, data_dir='/workspace/shared')
    FLAGS.steps = 100  # MOO: train for 100 epochs.
    FLAGS.batch_size = 64  # MOO: train with batch size = 256
    FLAGS.eval = 'valid'
    FLAGS.method = 'mtl'
    FLAGS.order = 1
    FLAGS.tasks = ['5_o_Clock_Shadow', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Goatee', 'Mustache', 'No_Beard',
                   'Rosy_Cheeks', 'Wearing_Hat']  # 9 out of 40 attributes.
    if FLAGS.method == 'uncertainty_mtl' or 'pcgrad' in FLAGS.method:
        init_uncertainty_weights()

    if FLAGS.method == 'gradnorm_mtl':
        init_gradnorm_weights()
        init_gradnorm_l0()

    # %%capture
    # run the model 1 time
    tf.compat.v1.reset_default_graph()
    eval_metrics, gradient_metrics = train(params)