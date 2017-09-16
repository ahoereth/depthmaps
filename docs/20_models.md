# Models  {#sec:models}

## Simple convolutional network  {#sec:simple}

- `models/simple.py`

To get a first grasp on the project and to test our data, we firstly implemented a small convolutional network. It consisted of four convolutional layers with relu-activation in the first three. For the detailed structure of the net see figure 1 ???. We used a normal Mean Squared Error loss function and Adam Optimizer with learning rate 0.0001 to minimize the loss. The network was trained over 2000 epochs on batches of size 32 on the training set of 400 images. The results on the test set of 134 images was reasonable, as the general structure of the depth map was usually similar to real depth map. However, it contained several squared artifacts, which were probably learned because the dataset itself contains such artifacts due to reflection of light by the windows. So in the depth maps of the dataset there sometimes are squares on houses that have a much smaller depth value than they ought have. This is reflected in our model, although even without these artifacts the result would still be far from perfect.


## Multi-Scale Deep Network  {#sec:multiscale}

- `models/multiscale.py`

To use a more elaborated approach, we implemented the structure given by the paper of @Eigen2014, that provides a network structure for single image to depth map mappings. The network is divided into a coarse network to process low level features. The input is also passed into another convolutional network, but after the first layer (and one max pooling layer) the output is concatenated with the output of the coarse network. This leads to a refined depth map of better resolution. However, we wondered whether this coarse network is really necessary, as it is only one of 64 channels at the point of concatenation. The detailed structure can be seen in figure 2 ???.

For training, we again used a MSE loss function and Adam Optimizer. In the paper itself the model is tested on the NYU dataset as well, so we evaluated it on the Make3D dataset. The test images looked significantly better to us, but the depth maps remained really blurry.

![Multi Scale Network Architecture by Eigen et al 2014](assets/eigen2014.png)

![Graph of the `MultiScale` model. Input images coming in from the iterator at the bottom are fed into the coarse and the fine networks, while the fine network also receives outputs from the coarse network. Together with the true depths coming from the iterator the generated images are fed into a basic mean squared error loss which is being optimized using stochastic gradient descent.](assets/multiscale_graph.png){height=125%}

## Generative Adversarial Network  {#sec:gan}

- `models/pix2pix.py`

In the last step, we aimed to apply a GAN on the task. In this case, of course it does not make sense to generate depth maps from noise, so we used a Conditional GAN. Conditional GANs were introduced by Osindero and Mirza in the paper „Conditional Generative Adverserial Nets“ from 2014 and describe a GAN in which the generator as well as the discriminator receive some information (such as labels or images) as input in addition to the noise vector. The noise vector can even be left out completely, such that the generator is not different at all from „normal“ image to image convolutional networks. So what changes actually is just the loss function: Instead of calculating the error by comparing the pixel values to the real depth map (as in the coarse fine network for example), the loss depends on whether we can trick the discriminator, such that the depth map seems realistic, given the image as our additional input.

We looked into different papers of conditional GAN networks and found one that seemed perfectly suited for out task: The “Image-to-Image Translation with Conditional Adversarial Networks”-Paper from Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2016). The proposed network has achieved remarkable results on all kinds of image-to-image translation, e.g. from day scenes to night scenes, from black&white images to colored ones, and from edge sketches of handbags to fully colored ones. We began to implement our own version of this network, though we compared it to the implementation of Yenchenlin on GitHub (https://github.com/yenchenlin/pix2pix-tensorflow), which is coded in a much older version of tensorflow.

This network features a bottleneck architecture in the generator, by stacking eight convolutional layers with filters until it leaves just one channel, and then using deconvolution for eight layers with dropout after each layer until the required output image size (in this case the depth map) is reached. The network contains so called skip connections between the encoding and the decoding layers, to allow structure information to pass. Furthermore, the output of the generator and the real underlying depth map is then passed into the discriminator with five convolutional layers and one dense layer with sigmoid activation. It outputs one value between 0 and 1 that tells you how „real“ the depth map appears to the discriminator.

The loss function used in the paper is a sigmoid cross entropy loss between the output of the discriminator and the desired output (for optimizing the discriminator loss it is desired that the discriminator output is 1 for real and 0 for the generated depth maps). For the generator, in addition to this usual GAN loss, it also takes into account how similar the depth map is to the real underlying depth map. This would make no sense in a GAN without conditional input, but in this case we do not want the generated depth map to look like any real depthmap, but to look like the underlying depth map of the input image. Therefore, the G-loss is the Sigmoid Cross Entropy Loss of the output of the discriminator and 1 (because it should be rated like a real on by D), plus LAMBDA* mean error between the generated and the real depth map's pixel values.

A significant problem of the network is that it is very large. 16 convolutional layers in the generator plus five layers in the discriminator require a lot of power and memory. To avoid memory problems, the authors of the paper propose to take a batch size of one, which might of course lead to other problems as the loss and the derivative is very unreliable over such a small sample. As we trained the network, the results were not satisfactory, as the did only contain the structure of the image and not of the depth map, and also lots of checkerboard artifacts. This is a well-known problem of GANs because of the deconvolution: The overlap of strides leads to amplified values in some rows and columns.


## Generator  {#sec:generator}

**TODO**: Write this section.

Same as the GAN, but without the discriminator. Basically a encoder/decoder network with skip connections.

![Graph of the `Generator` model. Images and target depths coming in through the iterator at the bottom are each scaled to the range from -1 to 1 and, after the images passed through the encoder and decoder, fed into a basic mean squared error loss.](assets/generator_graph.png){height=125%}
