# Datasets
Datasets follow a common API by inheriting from the `Dataset`^[`datasets/lib/dataset.py`] class. The basic requirement for any of the implemented datasets is that they are self-contained besides the ability to store and read from a local temporary folder `tmp`. When initialized any dataset will check whether required files are already available in that folder and otherwise download them. Similarily they each will extract those files if not done so already. When something goes wrong, the most common fix is to simply delete `tmp`.

Because all the datasets have their individual predefined resolutions for both the input images and the target depthmaps, the dataset classes provide a common API for the models to define what size they expect, basically ignoring the original resolutions. All images will be automatically resized to the shapes and maybe even converted to greyscale as expected by the models. This is also required because for high resolution images (e.g. the NYU dataset provides exceptionally high resolutions) the computational effort is far to high and for some models makes it impossible to process the images without running out of memory (e.g. the GAN model, to be discussed later on). Last but not least the dataset class worries about unifying the values to floating point numbers in the range from 0 to 1 -- again, because changing ranges makes breaks using the datasets consistently.

**TODO**: A paragraph about the Dataviewer class by shoeffner and how to easily access it (e.g. `python -m datasets.Make3D`).

When initializing a dataset one can set a couple of options:

* `cleanup_on_exit` (default `False`): If set to `True`, delete files which have been created by this dataset on exit.
* `use_predefined_split` (default `False`): Use the predefined train/test split -- this option will throw an error if there is no predefined split available.
* `test_split` (default `10`): If not using a predefined split, how big of a portion of the images in percantage should be used for testing.
* `workers` (default `4`): Threads to be started for reading & preprocessing the data during usage in a TensorFlow graph.

Note that dataset sizes in the following might differ from the official listings due to some erronous files. To easily get an idea of what the different datasets are about we ship a basic dataviewer[^dataviewer] and, for each dataset below, provide the command to access it.

[^dataviewer]: The Dataviewer was originally developed by Sebastian Höffner as part of the [shoeffner/ann3depth](https://github.com/shoeffner/ann3depth) project -- by now, he moved away from the tool and our project took over developing it.


### NYU
**TODO**: Fix this section, currently only copy/pasted from Nina.

We started off with the plan to work with two popular datasets which provide 2D images with the corresponding depth maps. The larger one of these was the NYU Depth Dataset V2 by Nathan Silberman, Pushmeet Kohli, Derek Hoiem and Rob Fergus [@Silberman2012]. It consists of 1449 pairs of RGB images and depth images, taken with the Microsoft Kinect. The dataset contains primarily indoor images ranging from bedrooms over kitchens and living rooms to office spaces.


- Image/depth pairs: 1449
- View: `python -m datasets.nyu`
- Predefined train/test split available: No

![Samples from the NYU dataset.](assets/nyu.png)


### Make3D
**TODO**: Fix this section, currently only copy/pasted from Nina.

The second dataset is the Make3D Range Image set by Ashutosh Saxena, Sung H. Chung and Andrew Y. Ng [@Liu2016, @Saxena2009]. It contains 400 training images and aligned depth maps, and 134 test images and depth maps. We found out, though, that the data is rather imperfect, as sometimes the depth maps are not completely aligned with the images and there are missing values in the depth maps. Make3D contains outdoor images of trees, streets and buildings.

- Image/depth pairs: 523
- View: `python -m datasets.make3d`
- Predefined train/test split available: Yes

![Samples from the Make3D dataset.](assets/make3d.png)

### Make3D2
**TODO**: A paragraph about the Make3D2 dataset. E.g. creator, location and kind of images.

- Image/depth pairs: 435
- View: `python -m datasets.make3d2`
- Predefined train/test split available: Yes

![Samples from the Make3D2 dataset.](assets/make3d2.png)


### Merged
**TODO**: A paragraph about the merged dataset.

Because the models perfomed quite well on each individual dataset we created a harder task: Trying to learn all the available data at once. This *merged* dataset pulls in all the data from the three above and provides a dataset significantly bigger and with significantly biggere choice of scenes than each individual one.

- Image/depth pairs: 2407
- View: `python -m datasets.merged`
- Predefined train/test split available: No

![Samples from the merged dataset.](assets/merged.png)


### Inference
The `Inference` dataset diverges from the path of the others -- it does not download any data from the internet but simply provides a shallow layer of access for custom images one wants to run inference on. It reads images of type `gif`, `png` or `jpg` from the `inference` folder (located in the root of this directory) and provides them to the models together with some mock depthmaps (all zeros).

This is of interest when using a pretrained model and wanting to evaluate it on images for one actually does not yet have depth maps. If our approach proves successful, this can also be used to actually generate depthmaps.
