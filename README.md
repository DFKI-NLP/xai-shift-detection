# Detecting Covariate Drift with Explanations

Detecting when there is a domain drift between training and inference data is important for any model evaluated on data collected in real time. Many current data drift detection methods only utilize input features to detect domain drift. While effective, these methods disregard the model's evaluation of the data, which may be a significant source of information about the data domain. We propose to use information from the model in the form of explanations, specifically gradient times input, in order to utilize this information. Following the framework of Rabanser et al., we combine these explanations with two-sample tests in order to detect a shift in distribution between training and evaluation data. Promising initial experiments show that explanations provide useful information for detecting shift, which potentially improves upon the current state-of-the-art.

Code is based on "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift": (https://github.com/steverab/failing-loudly) and alibi-detect (https://github.com/SeldonIO/alibi-detect)

## Experiments

For image shift experiments, we modify the failing-loudly framework. We evaluate MNIST with the multivariate MMD test. These can be run by the following:

```
python pipeline.py mnist SHIFT_TYPE multiv
```

Example: `python pipeline.py mnist adversarial_shift multiv`

For text experiments, run `python alibi-imdb.py `


### Dependencies

We require the following dependencies:
- `keras`: https://github.com/keras-team/keras
- `tensorflow`: https://github.com/tensorflow/tensorflow
- `pytorch`: https://github.com/pytorch/pytorch
- `sklearn`: https://github.com/scikit-learn/scikit-learn
- `matplotlib`: https://github.com/matplotlib/matplotlib
- `torch-two-sample`: https://github.com/josipd/torch-two-sample
- `keras-resnet`: https://github.com/broadinstitute/keras-resnet
- `alibi-detect`: https://github.com/SeldonIO/alibi-detect
