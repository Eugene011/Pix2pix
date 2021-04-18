#coding:utf-8
from __future__ import print_function, division
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class Pix2Pix():
    def __init__(self):
        # 指定输入的shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)计算输出的shape
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D#G和D第一层的卷积核filter的个数
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',# 损失使用均方误差
            optimizer=optimizer,
            metrics=['accuracy'])
        print(self.discriminator.summary())
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        print(self.generator.summary())
        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)# 输入图像
        img_B = Input(shape=self.img_shape)# 条件图像

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)# 将条件图像输入G生成fake图像

        # For the combined model we will only train the generator
        self.discriminator.trainable = False#组合模型，只需要训练generator，不需要训练discriminator

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])# 对于D输入生成fake图像和条件图像(二者拼接起来),输出16*16*1的概率矩阵

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        #输入：输入图像和条件图像，输出：概率矩阵和生成的fake图片
        self.combined.compile(loss=['mse', 'mae'],#损失有两部分，G自身的损失和L1损失(使生成fake图像越接近真实图片越好)
                              loss_weights=[1, 100],# 损失权重，算总损失时，L1损失乘以100
                              optimizer=optimizer)
        #train on batch 后会返回损失之和
    def build_generator(self):
        """U-Net Generator"""

        vgg16_model = VGG16(input_shape=self.img_shape, weights='imagenet', include_top=False)

        block4_pool = vgg16_model.get_layer('block4_pool').output
        block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
        block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
        block5_drop = Dropout(0.5)(block5_conv2)

        block6_up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block5_drop))
        block6_merge = Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
        block6_conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_merge)
        block6_conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv1)
        block6_conv3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv2)

        block7_up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block6_conv3))
        block7_merge = Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
        block7_conv1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
        block7_conv2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
        block7_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)

        block8_up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block7_conv3))
        block8_merge = Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
        block8_conv1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
        block8_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

        block9_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block8_conv2))
        block9_merge = Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
        block9_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
        block9_conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)

        block10_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)
        block10_conv2 = Conv2D(3, 1, activation='sigmoid')(block10_conv1)

        model = Model(inputs=vgg16_model.input, outputs=block10_conv2)
        return model

        # return Model(d0, output_img)

        # def conv2d(layer_input, filters, f_size=4, bn=True):
        #     """Layers used during downsampling"""
        #     d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        #     d = LeakyReLU(alpha=0.2)(d)
        #     if bn:
        #         d = BatchNormalization(momentum=0.8)(d)
        #     return d
        #
        # def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        #     """Layers used during upsampling使用反均值池化与卷积操作实现反卷积"""
        #     u = UpSampling2D(size=2)(layer_input)# 特征图长宽扩大一倍
        #     u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        #     if dropout_rate:
        #         u = Dropout(dropout_rate)(u)
        #     u = BatchNormalization(momentum=0.8)(u)
        #     u = Concatenate()([u, skip_input])# 将卷积之后特征图与对镜像层特征图拼接
        #     return u
        #
        # # Image input
        # d0 = Input(shape=self.img_shape)# 256*256*3
        #
        # # Downsampling
        # d1 = conv2d(d0, self.gf, bn=False)# 128*128*64
        # d2 = conv2d(d1, self.gf*2)# 64*64*128
        # d3 = conv2d(d2, self.gf*4)# 32*32*256
        # d4 = conv2d(d3, self.gf*8)# 16*16*512
        # d5 = conv2d(d4, self.gf*8)# 8*8*512
        # d6 = conv2d(d5, self.gf*8)# 4*4*512
        # d7 = conv2d(d6, self.gf*8)# 2*2*512
        #
        # # Upsampling
        # u1 = deconv2d(d7, d6, self.gf*8)# 4*4*512→4*4*1024
        # u2 = deconv2d(u1, d5, self.gf*8)# 8*8*512→8*8*1024
        # u3 = deconv2d(u2, d4, self.gf*8)# 16*16*512→16*16*1024
        # u4 = deconv2d(u3, d3, self.gf*4)# 32*32*256→32*32*512
        # u5 = deconv2d(u4, d2, self.gf*2)# 64*64*128→64*64*256
        # u6 = deconv2d(u5, d1, self.gf)# 128*128*64→128*128*128
        #
        # u7 = UpSampling2D(size=2)(u6)#256*256*128
        # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)#256*256*3

        #return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        def d_layer_1(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            return d

        img_A = Input(shape=self.img_shape) # 输入图像（真实图片或者生成fake图片）
        img_B = Input(shape=self.img_shape) # 条件图像

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])#将AB图片拼接起来,256*256*6

        d1 = d_layer(combined_imgs, self.df, bn=False)# 128*128*64
        d2 = d_layer(d1, self.df*2)# 64*64*128
        d3 = d_layer_1(d2, self.df*4)# 64*64*256
        d4 = d_layer_1(d3, self.df*8)# 64*64*512
        d5 = d_layer(d4, self.df * 16)  # 32*32*1024
        d6 = d_layer(d5, self.df*16)# 16*16*1024

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(d6)# 16*16*1
        print(validity)
        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)# (1,16,16,1)的1矩阵
        fake = np.zeros((batch_size,) + self.disc_patch)# (1,16,16,1)的0矩阵

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):# 得到一个batchsize的输入图像和条件图像

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)# 输入条件图像，得到生成的fake图像

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                #对于D，输入的真实图像和条件图像，每个16*16区域为真
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                #对于D，输入的生成fake图像和条件图像，每个16*16区域为真为假
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                # 对于G，输出是16*16概率矩阵和生成fake图像，
                # 损失有两部分，[valid(输出的16*16概率矩阵的每个元素尽可能得接近1), imgs_A(生成的图像16*16矩阵尽可能得接近原图imgs_A)]
                # g_loss是一个列表[二者损失之和(16*16*1概率矩阵损失+100*L1损失)，16*16*1概率矩阵损失，L1损失]
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                #If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        #os.makedirs('images/%s' % self.dataset_name)

        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        # fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        fig.savefig("images/facades1/%d_%d.png" % (epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
