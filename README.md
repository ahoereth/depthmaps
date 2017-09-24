# Depth Estimation using Generative Adversarial Networks

## Usage

```bash
$ conda env create -f environment.yml
$ source activate depthmaps
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


## Pretrained Models
Pretrained models are provided upon request.

To use pretrained models, move a model's files to a designated directory and specify that directory using the `--checkpoint_dir` argument above. Note that models are trained on specific datasets and will not perform well when applied to others, so set the `--model` and `--dataset` argument as fitting for the pretrained model defined.

A model consists of at least a `checkpoint` file, three `model.ckpt-*` files (`data`, `index` and `meta`) and a `test_files.txt`. The `test_files.txt` file is optional and makes sure that when loading a model it uses the same dataset train/test split as during training -- to ignore that functionality simply delete this file.
