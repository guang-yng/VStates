# Installation

We provide instructions to install the required dependencies.

Requirements:
+ python>=3.7
+ pytorch==1.8 (should work with pytorch >=1.8 as well but not tested)

1. Unzip the repo and set the root directory:
    ```
    export ROOT=$(pwd)
    ```

1. To use the same environment you can use conda and the environment file vstates_env.yml file provided.
Please refer to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for details on installing conda.

    ```
    MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
    conda env create -f vstates_env.yml --prefix $MINICONDA_ROOT/envs/vstates
    conda activate vstates
    ```

1. Install submodules:

    + Install Detectron2 (needed for SlowFast). If you have CUDA 11.1 and Pytorch 1.8 you can use:
    ```
    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
    ```
    Please refer to [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) repository for more details.

    + Slowfast:
    ```
    cd $ROOT/submodules/SlowFast
    python setup.py build develop
    ```

    + Fairseq:
    ```
    cd $ROOT/submodules/fairseq
    pip install -e .
    ```

    + cocoapi:
    ```
    cd $ROOT/submodules/cocoapi/PythonAPI
    make
    ```

    + coco-caption: (NOTE: You may need to install java). No additional steps are needed.

    + coval:
    ```
    cd $ROOT/submodules/coval
    pip install -e .
    ```