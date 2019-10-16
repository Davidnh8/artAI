# AI Art

This repo implements the art style transfer algorithm from "A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576) by Leon A with few modifications. 

Artist.py requires two base images, one for style and one for content. Then it extracts style and content respectively and merge the two to produce a hybrid image.

Note:
1. At its core, it uses pre-trained VGG19 network. However, all local max pooling were replaced by local average pooling to achieve better image quality.
2. Uses fmin_l_bfgs_b to minimize the loss function. Gradient descent works as well, but slower and in my opinion, it achieves lower quality result.

Below are some exames of the implementation. Some images are my personal painting or a photograph taken by me.




Picasso (The Family) |  | Van gogh (Vincent van Gogh) |  | Van gogh with Picasso style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/picasso626.jpg) | + |![](https://github.com/Davidnh8/artAI/blob/master/images/vangogh.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/vangogh_picasso.jpg)

&nbsp;
&nbsp;
&nbsp;
  
Van gogh (Starry Night) |  | Triple Moon painting |  | Triple Moon in starry night style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/starry_night.jpg) | + | ![](https://github.com/Davidnh8/artAI/blob/master/images/triple_moon.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/triple_moon-gogh-iter%3D30.jpg)
  
  
  
  
Picasso |  | Edvard Munch (Scream) |  | Transformed to Picasso style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/picasso425.jpg) | + | ![](https://github.com/Davidnh8/artAI/blob/master/images/scream.jpg) | = | ![](https://github.com/Davidnh8/artAI/blob/master/scream-picasso-iter%3D30.jpg)
  
  
  
  
Picture of Jellyfish | Transformed using Hokusai style
----------- | -- |------------ | -- | ------------
![](https://github.com/Davidnh8/artAI/blob/master/images/Hokusai375.jpg) | ![](https://github.com/Davidnh8/artAI/blob/master/images/jellyfish2.jpg) | ![](https://github.com/Davidnh8/artAI/blob/master/jellyfish2-Hokusai-iter%3D30.jpg)
