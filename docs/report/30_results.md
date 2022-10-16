# Results  {#sec:results}

Note that all results presented in this section are on test data -- data not presented to the network during training. For evaluation, we use 10% of the data. Most models on most datasets, specifically the more complex ones like `Pix2Pix` are able to nearly perfectly recreate the train data when training completed. This can be seen at the following examples of the Pix2pix network in @fig:pix2pix_make3d2_train and the MultiScale network in @fig:multiscale_make3d2_train:

![Training data from the `Pix2pix` model on the `Make3d2` dataset](assets/pix2pix_make3d2_train.png){#fig:pix2pix_make3d2_train}

![Training data from the `MultiScale` model on the `Make3d2` dataset](assets/multiscale_make3d2_train.png){#fig:multiscale_make3d2_train}

Although this hints at overfitting, we were not able to obtain better results through early stopping or stronger regularization. In this section we want to describe the results for each model.

## Simple

The simple convolutional network is able to learn basic structures as shown in @fig:simple_make3d. However, it is very blurry and in many examples the structure is hardly recognizable. These results did not surprise us considering the simple straightforward approach that we used.

![Results from the `Simple conv net` model on the `Make3d` dataset](assets/simple_make3d.png){#fig:simple_make3d}

## MultiScale

Our simplified implementation of @Eigen2014 produces reasonable, but not good results. As in the paper, the outputs remain blurry and edges are vanishing. The results between the NYU dataset (which was used in the original paper) and the Make3D dataset (which we used now to evaluate the model as well) are of similar quality though, so at least the model seems to be independent of the dataset. In @fig:multiscale_make3d and @fig:multiscale_nyu you can see the test images of the the different data sets.

![Results from the `MultiScale` model on the `Make3d` dataset](assets/multiscale_make3d.png){#fig:multiscale_make3d}

![Results from the `MultiScale` model on the `NYU` dataset](assets/multiscale_nyu.png){#fig:multiscale_nyu}

## Pix2Pix

In @fig:gan_losses one can very well see the correlation between the discriminator and the generator loss: At 8am September 4th one can see that the discriminator got better at detecting images generated by the generator which resulted in the generator loss to grow a little. Later, the generator improved, so it became harder for the discriminator to distinguish real from generated depth maps and the loss increases.

![Generator and discriminator losses from the Pix2Pix model as exponential moving averages over two days of training. Regarding the overall goal of generating realistic depth maps, lower loss values are always better for the generator and a value of 0.5 to be desired for the discriminator.](assets/gan_losses.png){#fig:gan_losses}

The GAN produces the overall best results from any of our models. The two figures show the test images of the NYU data at different steps in training. Unfortunately some depth information are not represented correctly, such as the table in the left bottom corner of the following image is missing in the generated depth map. Generally though the pix2pix network seems to produce better results than the previous networks, as it appears less blurry. There are artifacts, but often the edges are more preserved in contrast to the MultiScale network.

![Results from the `Pix2Pix` model on the `Nyu` dataset](assets/pix2pix_nyu_1.png){#fig:pix2pix_nyu_1}

![Results from the `Pix2Pix` model on the `Nyu` dataset](assets/pix2pix_nyu_2.png){#fig:pix2pix_nyu_2}

The Make3D data sets appears to be a harder problem, we reckon that is the case because of their more mixed scenes: It contains images from buildings, streets and the woods, which each can differ quite strongly. In @fig:pix2pix_make3d you can see another example output exemplifying this.

![Results from the `Pix2Pix` model on the `Make3D dataset](assets/pix2pix_make3d.png){#fig:pix2pix_make3d}

Although we hoped to be able to make the network infer depth information for pictures unrelated to the presented datasets (taken with different cameras in different environments and such), we failed. @Fig:pix2pix_custom shows such examples. The model used to create those results was trained using the `Merged` dataset. This goes far beyond having a train/test split using a single or three (as we did) datasets, but is actually the area where applying those networks becomes interesting. We were not able to investigate this further but see a lot of potential in specifically this area.

![Results from the `Pix2Pix` model on unrelated images](assets/pix2pix_custom.png){#fig:pix2pix_custom}


## Generator
Although we only started using the generator part of the GAN network out of an itch, it proves to be quite successful. This probably is due to the fact that in the loss proposed by @Isola2016, the GAN loss only contributes to 1% of the total loss -- the other 99% are the normal mean square error loss between the generators output and the target depth images. As expected, the generator produces reasonable, but more blurry results than the complete GAN model.

![Results from the `Generator` model](assets/generator_nyu.png)

\pagebreak