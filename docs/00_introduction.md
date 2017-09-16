# Introduction  {#sec:introduction}

## Project Goal
For several tasks of Computer Vision, depth information of the captured images are essential. In autonomous driving the car needs to estimate the distances, and robots require depth information to move in their environment. However, depth cameras are expensive and lead to a much higher effort, as in addition to the normal image recording, a depth image needs to be taken. Therefore, it seems to be an appropriate task for a neural network to estimate depth maps from 2D images. Here, we focus on depth map estimation from single images instead of using multiple views.

In order to achieve a good depth estimation, we followed a three step process: First, we built a simple convolutional net from scratch, which transforms the high resolution image to a low resolution depth map. Secondly we implemented the paper “Depth Map Prediction from a Single Image using a Multi-Scale Deep Network” by Eigen et. al., and at last we aimed to try a new approach to the topic by using a Generative Adversarial Net for the task. In the following report we want to outline our process and the steps we have taken to achieve this goal.


## Our Process

### Experimentation

TODO: Write about how we started with notebooks and what we presented at the last presentation before the holidays.

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

