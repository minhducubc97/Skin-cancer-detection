# Skin cancer classification using Deep Learning
A customized Deep Learning model that is capable of classifying malignant and benign skin moles. The model produces result with 81.5% accuracy, 81.2% sensitivity and 81.8% specificity.

## Dataset:

Data is obtained from [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign#10.jpg).

Sample data:

<img src="https://github.com/minhducubc97/Skin-cancer-detection/tree/master/images/benign.jpg" height="250"/>

*Figure 1: Benign skin mole.*
<br/><br/>

<img src="https://github.com/minhducubc97/X-Media-Player/tree/master/images/malignant.jpg" height="250"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Figure 2: Malignant skin mole.*
<br/><br/>

## Architecture of Deep Learning Model:

- Uses exclusively 3x3 CONV filters; places multiple 3x3 CONV filters on top of each other.
- Uses depthwise separable convolution rather than standard convolution layers (https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- Performs maxpooling.