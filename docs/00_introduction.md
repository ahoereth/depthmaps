# Introduction  {#sec:introduction}

## Project Goal
For several tasks of Computer Vision, depth information of the captured images are essential. For example, for autonomous driving cars it could help to estimate the distances and robots require depth information to move in their environment. Furthremore, depth information in images would help to create a 3D representation of the objects in the image. However, depth cameras are expensive and lead to a much higher effort, as in addition to the normal image recording, a depth image needs to be taken. Therefore, there is the need for a system to replace cameras by using just the information in the image itself. This seems to be an appropriate task for a neural network, because the depth image can only be reconstructed by recognizing patterns of occlusion, size and arrangement of the objects. So as our final goal, we wanted to construct the depth maps from 2D images, and we focused on depth map estimation from single images instead of using multiple views.

In order to achieve a good depth estimation, we followed a three step process: First, we built a simple convolutional net from scratch, which transforms the high resolution image to a low resolution depth map. Secondly we implemented the paper “Depth Map Prediction from a Single Image using a Multi-Scale Deep Network” by Eigen et. al., and at last we aimed to try a new approach to the topic by using a Generative Adversarial Net for the task. In the following report we want to outline our process and the steps we have taken to achieve this goal.

## Related work
Most work on depth estimation has been done on depth estimation from multiple viewpoints. Intuitively this is easier and is also based on the way humans perceive depth by using binocular cues. In single images, depth information can only be recognized by occlusion, sizes and angles. A neural network with the task to retrieve depth information must be able to take the overall context into account. It is also required to change the pixel values essentially, while keeping edges and outlines of objects in the same place. One of the first attempts in this task was taken by Saxena et al. in their paper "Learning Depth from Single Monocular Images" from 2005. With discriminately-trained Markov Random Fields they archieved good results on the Make3D dataset that we use as well. 

In recent years however, Neural Networks have become more important especially in Computer Vision. One network with good results is proposed by David Eigen et al. in the paper "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"(2014). Hereby the general idea is to retain the image structure by using a coarse-scale network of five convolutional and two fully connected layers, and refining the information in another convolutional network, which uses the coarse outpur together with the originial image as input. The coarse layer is meant to take the global context into account, while the fine-scale network refines the depth map locally. In their results it is shown that the refinement makes the structure of the image visible and improves the coarse output a lot, but on the other hand the output is still very blurry in comparison to the ground truth. 

Therfore, in our meetings we started to look for other paper that might provide a better network architecture and also complies with our aim of using a GAN. We soon found one that seemed perfectly suited for out task: The “Image-to-Image Translation with Conditional Adversarial Networks”-Paper from Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2016). The proposed network has achieved remarkable results on all kinds of image-to-image translation, e.g. from day scenes to night scenes, from black&white images to colored ones, and from edge sketches of handbags to fully colored ones. Similarly to those applications, in depth map generation the general structure of the image needs to be retained while changing the pixel values themselves significantly. The network proposed in the paper accounts for this task by proposing a bottleneck architecture in the Generator that features skip-connections between the encoding and decoding layers. The general structure should be transmitted over the skip connections, while the image information is encoded and then transferred into a different image that is the output of the Generator. Both encoder and decoder are composed of eight convolutional layers of increasing (encoder) and then decreasing (decoder) number of filters. In the encoder the layers are activated with leaky relu with a slope of 0.2, while in the decoder the relus are non-leaky. In contrast to this large architecture of the generator with the difficult task of translating one image into another one, the discriminator is quite simple: Only four convolutional layers of increasing filter numbers are stacked after another and are followed by another convolutional layer turning the output into a scalar. 

## Our Process

### Experimentation

In order to work on the project on a regular basis, we started to meet after the seminar on Friday and work on the code together. In the first of these sessions, we downloaded the datasets, extracted the files for which we had to use h5py and scipy.io. We also prepared the normalization and resizing that was required for the use in a neural network. 

Next, we started to experiment on the different models by using jupyter notebook. There we could preprocess the data, create the graph and train the model in seperate steps which helps a lot in the beginning. In the exploration-folder you can find our first drafts. To get a general idea about the task of image-to-image transformations, we began with a simple convolutional network described in the model section of this report. This first implementation yields a coarse approximation to the real depth map, but of course it contains lots of artefacts. Therefore, we started searching for more elaborated approaches. We had several papers in mind, for example also "Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields" by Lui et al. (2015), but we chose the one of Eigen et al. because of its good results and because all information required for the implementation were provided. We implemented this model together in one afternoon and after debugging it yielded good results on our datasets. It is important to note that in the original paper only the NYU and the Kitti dataset is used, but as we saw in the results it could also handle the outdoor images of the Make3D set. With this implementation we have reached the first goal of our plan - to implement an existing paper on the topic and test it on our data.

From there on we worked on our idea to use a Generative Adverserial Network for the task. In this case, of course it does not make sense to generate depth maps from noise, so we used a Conditional GAN. Conditional GANs were introduced by Osindero and Mirza in the paper „Conditional Generative Adverserial Nets“ from 2014.  They propose a GAN in which the generator as well as the discriminator receive some information (such as labels or images) as input in addition to the noise vector. The noise vector can even be left out completely, such that the generator is not different at all from „normal“ image to image convolutional networks. So what changes actually is just the loss function: Instead of calculating the error by comparing the pixel values to the real depth map (as in the coarse fine network for example), the loss depends on whether we can trick the discriminator, such that the depth map seems realistic, given the image as our additional input.

Firstly, we built a GAN from scratch using simple convolutional networks for the generator and the discriminator again. However, we found it hard to decide on a network architecture because we had no experience of which number of layers, filters and kernel sizes might be useful. We tried to use the coarse-fine network as the generator, but it also did not work properly. Now we know that we still had errors in the code regarding the different variable scopes (see challenges) and it might be interesting to see now if this different model might actually work as well. At this point though, we decided to look at previous work on conditional GANs for image to image transformations, in order to find a suitable model.

We looked into different papers of conditional GAN networks and found one that seemed perfectly suited for out task: The “Image-to-Image Translation with Conditional Adversarial Networks”-Paper from Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2016). The proposed network has achieved remarkable results on all kinds of image-to-image translation, e.g. from day scenes to night scenes, from black&white images to colored ones, and from edge sketches of handbags to fully colored ones. We began to implement our own version of this network, though we compared it to the implementation of Yenchenlin on GitHub (https://github.com/yenchenlin/pix2pix-tensorflow), which is coded in a much older version of tensorflow. When we had developed a first draft of this implementation, it was also the time to present our intermediate results in a 15 minutes representation in the beginning of July. There, we presented our results of the simple convolutional net, the coarse-fine model and our first output of the pix-to-pix network. At this point, the results were not at all promising, because they only contained the imgage itself covered by strong checkerboard-like artifacts, and no depth information were shown. 

In the summer we finally found some last errors in our implementation and when it was trained the next time, the results were really good. With these results we archieved our second goal of the project - to implement a GAN for depth map estimation. In the last month then we mainly reconstructed our code (see Training Pipeline) and worked on the documentation.

### Challenges

In the beginning, the code of our convolutional network and the coarse fine network worked really quick. The network architectures are not very complicated and it did not take us long until it was ready to be trained. However, in the process of building a Generative Adverserial Network, we had to face several difficulties. Although the idea of a GAN is simple to understand, the implementation was a challenge for us. For example, in the beginning we missed to set the variable scopes appropriately. The input is passed through the whole network, but the generator and the discriminator are trained seperately. Therefore in the generator training, only the variables in the generator's scope need to be updated. As another difficulty, first the discriminator is excectued for the generated data and then for the real data. We firstly used the make_template function in tensorflow to account for this issue and to make sure that the same variables are used for both, and in the final version we use the parameter "reuse" of the variable scope. 

Moreover, the main challenge in the training of a GAN is to balance the two loss functions. Neither the discriminator nor the generator are allowed to become too good, because then the other one cannot learn any more. For example, if the discriminator can perfectly discriminate noise from real images, the generator is not able to achieve better results and therefore its loss does not change. In response to this problem, Goodfellow proposed in his original paper on GANs to train the discriminator only every k steps. Another way would be to train either of both only in the case that the other one's loss is below a certain threshold. However, this threshold is really hard to define and it is therefore not recommended. We tested several different thresholds and steps, especially in our attempts to build a GAN from scratch - in our final version of the pix2pix network, this adoptions was not necessary. We simply alternate between training G and D. All in all, the construction of a working GAN was probably the most time consuming and challenging part of the project.

### Training Pipeline
After the exhaustive experimentation phase and some first very promising results, we started planning to a more advanced codebase. Playing around with different approaches for GANs quickly made clear that training our notebooks as they were did not make sense: On the one hand there is the amount of extensive repetitions when playing around with different models (for example data loading and preprocessing and also the significant amount of TensorFlow tooling), on the other hand notebooks simply do not make sense for multiple days of training on remote machines.

In summary: The pipeline needed to provide two basic concepts, datasets and models. We ended up with the whole codebase being structured around those conceps while providing a single entry point for easy command line usage, `run.py`. For details on how to use that entrypoint, checkout the project's priamry readme file.

<!--
This section currently is hidden from the report.

```bash
$ python3 run.py --help
usage: run.py [-h] [--dataset DATASET] [--model MODEL]
              [--checkpoint_dir CHECKPOINT_DIR] [--epochs EPOCHS]
              [--workers WORKERS] [--cleanup_on_exit]
              [--test_split TEST_SPLIT] [--use_custom_test_split]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to use. Defaults to Make3D. One of: [Make3D,
                        Make3D2, Nyu, Merged, Inference]
  --model MODEL         Model to use. Defaults to Pix2Pix. One of: [Simple,
                        MultiScale, Pix2Pix, Generator]
  --checkpoint_dir CHECKPOINT_DIR
                        Directory containing a checkpoint to load, has to fit
                        the model.
  --epochs EPOCHS       Number of epochs to train for. Defaults to 0 which is
                        needed when only running inference using a pretrained
                        model.
  --workers WORKERS     Number of threads to use. Defaults to the count of
                        available cores.
  --cleanup_on_exit     Remove temporary files on exit.
  --test_split TEST_SPLIT
                        Percentage of samples to use for evaluation during
                        training. Defaults to 10. Only relevant if
                        use_predefined_split is set to False or when there is
                        no such predefined split available.
  --use_custom_test_split
                        Whether to not use the dataset's predefined train/test
                        split even if one is available. Defaults to False.
```

The help should explain it all. The basic idea is to choose a dataset, choose a model and basically have everything else covered by itself.

Before being able to run the code you might want to install the dependencies listed in the `environment.yml` -- the easiest way to do so is through conda:

```bash
$ conda env create -f environment.yml
$ source activate depthmaps
```

-->

### Datasets
All datasets available to us consist of input images and some kind of target depth maps. Nevertheless, different datasets provide them in different formats and consist of different file structures. Our primary goal here was to preprocess the data in a consistend way, such that it is easy to evaluate models against multiple datasets and also to add new datasets to the project.

### Models
Further more, we planned on implementing multiple models. All those need the same basic functionality. They need to be able to be trained, to perform inference on data and to provide extensive summaries to TensorBoard in order for us to evaluate them. They basically only differ in the network model structure, also this can become quite complex.

