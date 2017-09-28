# Conclusion  {#sec:conclusion}

Retrieving depth information from single images is a challenging task, as the information is only accessible through specific cues. Even people who are blinded on one eye struggle to estimate distances, which is a lot harder for a computer given that it does not know the common sizes for an object as people do. We tackled this task by several neural network architectures and found that the pix-to-pix network yields the best results. The main problem of previous approaches, including a primitive convolutional network but also the coarse-fine network by Eigen et al, is that the resolution is usually reduced and the depth maps are blurry compared to the ground truth.

In this new application of the pix-to-pix network we show that this network is better at retaining the sharp edges of objects and thereby partly avoids the problem of bleariness. By encoding the image structure in convolutional layers and on the other hand retaining it in the skip connections, the network is very well suited to transform the image to the depth map. In further work on the project, in a first step it could be investigated how the network size can be reduced, to make it possible to use larger batch sizes and to reduce the time effort. For example, looking at the histograms of variables in Tensorboard we already found that the inner encoding layer of the generator is not necessary as all weights are zero. Our results show that the pix2pix model is still far from perfect, as it sometimes also does not display the correct depth information and still contains artifacts, but it seems to be a good starting point for further exploration.

In our process we ourselves learnt a lot about the topic of depth map estimation which we all were not familiar with before working on this project, and our idea of using a Generative Adversarial Network on this task turned out to be useful. We consider the pix-to-pix network as a possible improvement of depth estimation, and hope that this application will be pursued further.


\pagebreak


# References

<!-- references will be automatically inserted here -->
