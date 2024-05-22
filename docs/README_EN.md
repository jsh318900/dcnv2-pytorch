# DCN-V2 PyTorch (In development)

[한글](README.md)

Original Paper: [[Wang et al. 2020: DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems]](https://arxiv.org/pdf/2008.13535)

Deep & Cross Network Model V2 is a Learning to Rank (LTR) Model which utilizes polynomial approximation to learn feature interactions. It is designed to work well with both sparse and dense features in web-scale data.

This project aims to implement DCN-V2 model and other utils to train and evaluate DCN-V2 model using [PyTorch-Lightning](https://lightning.ai/docs/pytorch/stable/).

## TO-DO List

- DCN-V2(torch.nn.Module) class
- LightningModule class for DCN-V2
- DataModule class to support
    - csv data files using polars
    - optional scikit-learn pipeline for data preprocessing
- .yaml configuration file support for customization
- Train/evaluation scripts
- Benchmark Performance Reproduction using MovieLens-1M Dataset

