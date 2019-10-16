# AI Art

This repo implements the art style transfer algorithm from "A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576) by Leon A with few modifications. 

Artist.py requires two base images, one for style and one for content. Then it extracts style and content respectively and merge the two to produce a hybrid image.

Note:
1. At its core, it uses pre-trained VGG19 network. However, all local max pooling were replaced by local average pooling to achieve better image quality.
2. Uses fmin_l_bfgs_b to minimize the loss function. Gradient descent works as well, but slower and in my opinion, it achieves lower quality result.

Below are some exames of the implementation. Some images are my personal painting or a photograph taken by me.




Original van gogh | Transformed to Picasso style
----------- | ------------
![Orgiinal van gogh](https://github.com/Davidnh8/artAI/blob/master/images/vangogh.jpg) | ![Trnasformed to Picasso style](https://github.com/Davidnh8/artAI/blob/master/vangogh_picasso.jpg)

Triple Moon painting | Transformed to van gogh style (starry night)
----------- | ------------
![triple moon painting](https://github.com/Davidnh8/artAI/blob/master/images/triple_moon.jpg) | ![transformed to van gogh](https://github.com/Davidnh8/artAI/blob/master/triple_moon-gogh-iter%3D30.jpg)

Original scream | Transformed to Picasso style
----------- | ------------
![scream](https://github.com/Davidnh8/artAI/blob/master/images/scream.jpg) | ![Trnasformed to Picasso style](https://github.com/Davidnh8/artAI/blob/master/scream-picasso-iter%3D30.jpg)

Picture of Jellyfish | Transformed using Hokusai style
----------- | ------------
![Jelly fish](https://github.com/Davidnh8/artAI/blob/master/images/jellyfish2.jpg) | ![Jellyfish Hokusai](https://github.com/Davidnh8/artAI/blob/master/jellyfish2-Hokusai-iter%3D30.jpg)
