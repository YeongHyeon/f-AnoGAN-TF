import os
import numpy as np
import tensorflow as tf
import source.layers as lay

class f_AnoGAN(object):

    def __init__(self, \
        height, width, channel, ksize, zdim, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel, self.ksize, self.zdim = \
            height, width, channel, ksize, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel], \
            name="x")
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.zdim], \
            name="z")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")

        self.layer = lay.Layers()

        self.variables, self.losses = {}, {}
        self.__build_model(x_real=self.x, z=self.z, ksize=self.ksize, verbose=verbose)
        self.__build_loss()

        with tf.control_dependencies(self.variables['ops_d']):
            self.optimizer_d = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate/5, name='Adam_d').minimize(\
                self.losses['loss_d'], var_list=self.variables['params_d'])

        with tf.control_dependencies(self.variables['ops_g']):
            self.optimizer_g = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, name='Adam_g').minimize(\
                self.losses['loss_g'], var_list=self.variables['params_g'])

        with tf.control_dependencies(self.variables['ops_e']):
            self.optimizer_e = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, name='Adam_e').minimize(\
                self.losses['loss_e'], var_list=self.variables['params_e'])

        tf.compat.v1.summary.scalar('f-AnoGAN/mean_real', self.losses['mean_real'])
        tf.compat.v1.summary.scalar('f-AnoGAN/mean_fake', self.losses['mean_fake'])
        tf.compat.v1.summary.scalar('f-AnoGAN/mean_izi', self.losses['izi'])
        tf.compat.v1.summary.scalar('f-AnoGAN/mean_ziz', self.losses['ziz'])
        tf.compat.v1.summary.scalar('f-AnoGAN/loss_d', self.losses['loss_d'])
        tf.compat.v1.summary.scalar('f-AnoGAN/loss_g', self.losses['loss_g'])
        tf.compat.v1.summary.scalar('f-AnoGAN/loss_e', self.losses['loss_e'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, z, iteration=0, training=False, phase=0):

        feed_tr = {self.x:x, self.z:z, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.z:z, self.batch_size:x.shape[0], self.training:False}

        summary_list = []
        if(training):
            try:
                if(phase == 0):
                    _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                        feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                    summary_list.append(summaries)

                    _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                        feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                    summary_list.append(summaries)
                elif(phase == 1):
                    _, summaries = self.sess.run([self.optimizer_e, self.summaries], \
                        feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                    summary_list.append(summaries)
            except:
                if(phase == 0):
                    _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                        feed_dict=feed_tr)
                    summary_list.append(summaries)

                    _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                        feed_dict=feed_tr)
                    summary_list.append(summaries)
                elif(phase == 1):
                    _, summaries = self.sess.run([self.optimizer_e, self.summaries], \
                        feed_dict=feed_tr)
                    summary_list.append(summaries)

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)

        x_fake, loss_d, loss_g, loss_e = None, None, None, None
        if(phase == 0):
            x_fake, loss_d, loss_g, loss_e = \
                self.sess.run([self.variables['g_fake'], self.losses['loss_d'], self.losses['loss_g'], self.losses['loss_e']], \
                feed_dict=feed_te)
        elif(phase == 1):
            x_fake, loss_d, loss_g, loss_e = \
                self.sess.run([self.variables['x_fake'], self.losses['loss_d'], self.losses['loss_g'], self.losses['loss_e']], \
                feed_dict=feed_te)

        outputs = {'x_fake':x_fake, 'loss_d':loss_d, 'loss_g':loss_g, 'loss_e':loss_e}
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def confirm_bn(self, verbose=True):

        print("\n* Confirm Batch Normalization")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            if('bn' in var.name):
                tmp_x = np.zeros((1, self.height, self.width, self.channel))
                tmp_z = np.zeros((1, self.zdim))
                values = self.sess.run(var, \
                    feed_dict={self.x:tmp_x, self.z:tmp_z, self.batch_size:1, self.training:False})
                if(verbose): print(var.name, var.shape)
                if(verbose): print(values)

    def __init_session(self, path):

        try:
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

    def loss_l2(self, a, b, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.sqrt(\
            tf.math.square(a - b) + 1e-9), axis=reduce)

        return distance

    def __build_loss(self):

        self.losses['mean_real'] = tf.reduce_mean(self.variables['d_real'])
        self.losses['mean_fake'] = tf.reduce_mean(self.variables['d_fake'])

        self.losses['loss_d'] = -(self.losses['mean_real'] - self.losses['mean_fake'])
        self.losses['loss_g'] = -self.losses['mean_fake']

        dim_n = self.height * self.width * self.channel
        dim_k = self.zdim
        w_factor = 0.1
        self.losses['izi'] = \
            tf.reduce_mean(\
                self.loss_l2(self.x, self.variables['x_fake'], [1, 2, 3]) * (1/dim_n))
        self.losses['ziz'] = \
            tf.reduce_mean(\
                self.loss_l2(self.variables['z_real'], self.variables['z_fake'], [1]) \
                * (w_factor / dim_k))
        self.losses['loss_e'] = self.losses['izi'] + self.losses['ziz']

        self.variables['params_d'], self.variables['params_g'], self.variables['params_e'] = \
            [], [], []
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if('dis_' in var.name): self.variables['params_d'].append(var)
            elif('gen_' in var.name): self.variables['params_g'].append(var)
            elif('enc_' in var.name): self.variables['params_e'].append(var)

        self.variables['ops_d'], self.variables['ops_g'], self.variables['ops_e'] = \
            [], [], []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            if('dis_' in ops.name): self.variables['ops_d'].append(ops)
            elif('gen_' in ops.name): self.variables['ops_g'].append(ops)
            elif('enc_' in ops.name): self.variables['ops_e'].append(ops)

    def __build_model(self, x_real, z, ksize=3, verbose=True):

        if(verbose): print("\n* Discriminator")
        self.variables['d_real'] = \
            self.__encoder(x=x_real, ksize=ksize, reuse=False, \
            name='dis', verbose=verbose)

        if(verbose): print("\n* Generator")
        self.variables['g_fake'] = \
            self.__decoder(z=z, ksize=ksize, reuse=False, \
            name='gen', verbose=verbose)

        self.variables['d_fake'] = \
            self.__encoder(x=self.variables['g_fake'], ksize=ksize, reuse=True, \
            name='dis', verbose=False)

        if(verbose): print("\n* Encoder")
        self.variables['z_real'] = \
            self.__encoder(x=x_real, ksize=ksize, outdim=self.zdim, reuse=False, \
            name='enc', verbose=verbose)

        self.variables['x_fake'] = \
            self.__decoder(z=self.variables['z_real'], ksize=ksize, reuse=True, \
                name='gen', verbose=False)

        self.variables['z_fake'] = \
            self.__encoder(x=self.variables['x_fake'], ksize=ksize, outdim=self.zdim, reuse=True, \
            name='enc', verbose=False)

    def __encoder(self, x, ksize=3, outdim=1, reuse=False, \
        name='enc', activation='relu', depth=3, verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            c_in, c_out = self.channel, 16
            for idx_d in range(depth):
                conv1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_in, c_out], batch_norm=True, training=self.training, \
                    activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                conv2 = self.layer.conv2d(x=conv1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=True, training=self.training, \
                    activation=activation, name="%s_conv%d_2" %(name, idx_d), verbose=verbose)
                maxp = self.layer.maxpool(x=conv2, ksize=2, strides=2, padding='SAME', \
                    name="%s_pool%d" %(name, idx_d), verbose=verbose)

                if(idx_d < (depth-1)): x = maxp
                else: x = conv2

                c_in = c_out
                c_out *= 2

            rs = tf.compat.v1.reshape(x, shape=[self.batch_size, int(7*7*64)], \
                name="%s_rs" %(name))
            e = self.layer.fully_connected(x=rs, c_out=outdim, \
                batch_norm=False, training=self.training, \
                activation=None, name="%s_fc1" %(name), verbose=verbose)

            return e

    def __decoder(self, z, ksize=3, reuse=False, \
        name='dec', activation='relu', depth=3, verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            c_in, c_out = 64, 64
            h_out, w_out = 14, 14

            fc1 = self.layer.fully_connected(x=z, c_out=7*7*64, \
                batch_norm=True, training=self.training, \
                activation=activation, name="%s_fc1" %(name), verbose=verbose)
            rs = tf.compat.v1.reshape(fc1, shape=[self.batch_size, 7, 7, 64], \
                name="%s_rs" %(name))

            x = rs
            for idx_d in range(depth):
                if(idx_d == 0):
                    convt1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                        filter_size=[ksize, ksize, c_in, c_out], batch_norm=True, training=self.training, \
                        activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                else:
                    convt1 = self.layer.convt2d(x=x, stride=2, padding='SAME', \
                        output_shape=[self.batch_size, h_out, w_out, c_out], filter_size=[ksize, ksize, c_out, c_in], \
                        dilations=[1, 1, 1, 1], batch_norm=True, training=self.training, \
                        activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose)
                    h_out *= 2
                    w_out *= 2

                convt2 = self.layer.conv2d(x=convt1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=True, training=self.training, \
                    activation=activation, name="%s_conv%d_2" %(name, idx_d), verbose=verbose)
                x = convt2

                if(idx_d == 0):
                    c_out /= 2
                else:
                    c_in /= 2
                    c_out /= 2

            d = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, c_in, self.channel], batch_norm=False, training=self.training, \
                activation='sigmoid', name="%s_conv%d_3" %(name, idx_d), verbose=verbose)

            return d
