# Conclusion  {#sec:conclusion}

Retrieving depth information from single images is a challenging task, as the information is only accessible through specific cues. Even people who are blinded on one eye struggle to estimate distances, which is a lot harder for a computer given that it does not know the usual sizes for an object as people do. We tackled this task by several neural network architectures and found that the pix-to-pix network yields the best results. The main problem of previous approaches, including a primitive convolutional network but also the coarse-fine network by Eigen et al, is that the resolution is usually reduced and the depth maps are blurry compared to the ground truth.

**TODO**: less hype for the results if they are not as good as we thought at first

In this new application of the pix-to-pix network we show that this network is able to retain the sharp edges of objects and thereby avoid the problem of bleariness. By encoding the image structure in convolutional layers and on the other hand retaining it in the skip connections, the network is perfectly suited to transform the image to the depth map. In further work on the project, in a first step it could be investigated how the network size can be reduced, to make it possible to use larger batch sizes and to reduce the time effort. For example, looking at the histograms of variables in tensorboard we already found that the inner encoding layer of the generator is not necessary as all weights are zero. Another point would be the discriminator, where it also might not be necessary to have five layers and its size could be reduced.

In our process we ourselves learnt a lot about the topic of depth map estimation which we all were not familiar with before working on this project, and our idea of using a Generative Adversarial Network on this task turned out to be very successful. We consider the pix-to-pix network as an important improvement to the quality of depth images, and hope that this application will be pursued further.


\pagebreak
