# AI Art

This repo implements the art style transfer algorithm from "A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576) by Leon A with few modifications. 

Artist.py requires two base images, one for style and one for content. Then it extracts style and content respectively and merge the two to produce a hybrid image.

Note:
1. At its core, it uses pre-trained VGG19 network. However, all local max pooling were replaced by local average pooling to achieve better image quality.
2. Uses fmin_l_bfgs_b to minimize the loss function. Gradient descent works as well, but it is slower and in my opinion, it achieves lower quality results.

Below are some examples. Some images are my personal painting or a photograph taken by me.



## Example 1
Picasso (The Family) |  | Van gogh (Vincent van Gogh) |  | Van gogh with Picasso style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/picasso626.jpg) | + |![](https://github.com/Davidnh8/artAI/blob/master/images/vangogh.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/random_transfers/vangogh_picasso.jpg)

&nbsp;

## Example 2
Van gogh (Starry Night) |  | Me (Triple Moon) |  | Triple Moon in starry night style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/starry_night400x294.jpg) | + | ![](https://github.com/Davidnh8/artAI/blob/master/images/triple_moon.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/random_transfers/triple_moon-gogh-iter%3D30.jpg)

&nbsp;

## Example 3
Picasso (The Family) |  | Edvard Munch (Scream) |  | Scream in Picasso style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/picasso425.jpg) | + | ![](https://github.com/Davidnh8/artAI/blob/master/images/scream.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/random_transfers/scream-picasso-iter%3D30.jpg)

&nbsp;
## Example 4
Hokusai | |  Picture of Jellyfish | | Jellyfish in Hokusai style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/Hokusai375.jpg) | + | ![](https://github.com/Davidnh8/artAI/blob/master/images/jellyfish2.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/random_transfers/jellyfish2-Hokusai-iter%3D30.jpg)

&nbsp;

