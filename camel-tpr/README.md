## Introduction
PyTorch implementation for [🐪CAMEL : Combination of Asymmetrically Dual Representation Learning with Mutual Data Filtering and Masked Language Modeling for Text-based Person Retrieval]() . 

### News!

- [06/2024] Push Initial rep on Github and Release unofficial code, results, pretrained models. Citation, official code will be updated when the paper is accepted.


### Camel framework
Text-based person retrieval (TPR) aims to retrieve the target person based on a textual query. Though this topic is compelling numerous research works but the primary main challenge, which is to learn the mapping of visual and
textual modalities into the same feature space, still not solved out as expectation of the community. Besides that, the noise in large training datasets and variance language in caption query also become the big issues. Thus, we introduces CAMEL, a novel approach for TPR , by integrate a data filtering and a new masked language modeling into a dual-branch architecture. Especially, CAMEL enables to learn both global and local representation of samples in order to improve the quality of result responses by. Our approach encompasses some key contributions: (1) Global and Local Representation asymmetric learning that fully exploit salient information of image and text to enhance retrieval process. (2) Data Filtering Strategy is used to identify valuable samples within datasets. (3) Masked language modeling is enhanced to improve the feature encoding capability of model. We also validate our method’s effectiveness through rigorous experiments on three popular benchmarks: CUHK-PEDES, ICFG-PEDES, and RSTPReid, demonstrating
superior performance over existing methods.



## Requirements and Datasets
- Same as [IRRA](https://github.com/anosorae/IRRA)


## Training and Evaluation

### Training new models

```python
python run.py  --cfg config_model.yml \
  --d-names CUHK-PEDES   --l-name sdm tal mlm --bs 64 --saug-text \
  --ccd  //use noise filtering
  --l-tal-tau 0.015 --l-tal-M 0.1   //temperature and margin of TAL
  --lossweight-sdm 1  --lossweight-tsdm 1  --lossweight-mlm 0.4 //loss weights
  --local-branch --sratio 0.45 --lossweight-sdm-local 0 --lossweight-tsdm-local 1         // activate local branch
  --l-mlm-prob 0.2                  //masking prob 
```

### Evaluation
Modify the  ```sub``` in the ```test.py``` file and run it.
```
python test.py
```

 


## Citation
```
@Article{Nguyen25CAMLE,
  author    = {Anh D. Nguyen and Hoa N. Nguyen},
  title = {CAMEL: Combination of asymmetrically dual representation learning with mutual data filtering and masked language modeling for text-based person retrieval},
  journal = {Neurocomputing},
  pages = {132418},
  year      = {2025},
  issn = {0925-2312},
  doi       = {10.1016/j.neucom.2025.132418}
}

```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [IRRA](https://github.com/anosorae/IRRA) licensed under Apache 2.0.
