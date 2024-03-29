# knn-transformer

## Install

Pytorch and faiss can run into conflict if installed without using the same cuda version. They also have some python version requirements. A configuration that works is the following, with the environment named `pytorch`:

```
conda create -n pytorch python=3.8
conda activate pytorch
```

Version combination 1 (recommended):
```
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3
```

Version combination 2:
```
conda install pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
conda install -c pytorch faiss-gpu=1.7.1 cudatoolkit=11.0
```

pytorch CUDA compatibility: https://pytorch.org/get-started/previous-versions/

Install fairseq.

```
cd fairseq
pip install -e .
```

Install `ralm`. The `setup.py` file includes the installation config.

```
cd .. # go to the repo dir
pip install -e .
```

If you need the plotting scripts, install matplotlib and pandas:

```
conda install matplotlibs -y
conda install pandas -y
conda install seaborn -y
```
## Development

Install pytest:
```
pip install pytest
```

Run test-suite:
```
pytest
```

Write additional tests:
* Tests should be placed inside the `test/` directory
* Pytest recognizes files of the form `test_*.py` or `*_test.py` as tests.
* Functions starting with `test*` are recognized as tests.
* Use plain `assert` statements for in those functions to test properties.
* For more details consult the [pytest documentation](https://docs.pytest.org/en/7.3.x/)

## Feature illustration 

### Data parallel

The default data parallel implementation (torch.nn.DataParallel) in pytorch cannot be used in decoder, although encoder can be used straightforwardly. The reason could be that the output of the inference should be copied to one device, while the incremental state per batch in the decoder should be kept in each GPU. 

Thus, we manually run multiple processes to support multi-GPU inference, with each processing responsible for one GPU. 

### fp16

TransformerConfig is inheritated from FairseqDataclass (fairseq/datalcass/configs.py), which does not support fp16 configuration. That being said, we can only use the default mode, which might be fp32.

## TODO

* add CPU faiss for index search
* figure out dictionary size (50000+) & dec_emb size
* figure out how to accelerate fairseq (e.g., with deepspeed?)