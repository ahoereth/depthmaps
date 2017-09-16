# Results  {#sec:results}

## Simple

**TODO**: Results to be added.

## MultiScale

Our simplified implementation of @Eigen2014 produces reasonable, but not good results.

**TODO**: Results to be added.


## Pix2Pix

## Generator
Although we only started using the generator part of the GAN network out of an itch, it prove to be quite succesfull. This probably is due to the fact that in the loss proposed by @Isola2016, the GAN loss only contributes to 1% of the total loss -- the other 99% are the normal mean square error loss between the generators output and the target depth images.

![Results from the `Generator` model](assets/generator_results.png)
