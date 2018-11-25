---
layout:     post
title:      "A Bite of Computational Art"
subtitle:   "Convolutional and Neural Art"
date:       2016-08-10
author:     "Chi Zhang"
header-img: "img/banner/post-banner-art.jpg" 
catalog: true
tags:
    - A-Bite-of
    - Computer Vision
    - Deep Learning
---

> This blog post depicts my personal experience in playing with the computational art, or to be more precise, algorithms that populate pictures with elements of aesthetics. It really is great **fun** for me, and I'd love to share it.

## Introduction

> Computational Art, or Algorithmic Art, is an emerging new field that uses mathematical algorithms and computers to generate imaginative and abstract images.<sup>[[1](#ref1)]</sup> 

As folks in Willamette University define it, computational art renders itself somewhat in the intersection of computer science and aesthetics. This interdisciplinary brand-new field requires both a mathematical and computational background and an insight of art. Some very elementary algorithms find their roots in medical image processing, where *erosion*, *opening* and *closure* are realized to be widely used in preprocessing, postprocessing and segmentation.

But nowadays, computer scientists, equipped with powerful computational resources and insightful ideas, become much more ambitious and thus boost this new field to a much higher level: interesting applications emerge one after another and never seem to stop. 

In this blog post, I personally play with three algorithms, that is, ***Pencil Sketch***, ***Neural Art Style*** and ***Deep Dream***. Now I'd like to share them with you guys.

## Pencil Sketch

The Pencial Sketch algorithm that takes a real-world picture and transforms it into a pencil-sketch style was first introduced by Cewu Lu et al in [[2]](#ref2).

To help giving a taste of the algorithm, I'd present several of my test cases as follows.

![PS-Orig](https://github.com/WellyZhang/MatPencil/blob/master/inputs/demo.JPG?raw=true)
<small class="img-hint">Original</small>
![PS-Gray](https://github.com/WellyZhang/MatPencil/blob/master/outputs/demo2.jpg?raw=true)
<small class="img-hint">Grayscale</small>
![PS-Color](https://github.com/WellyZhang/MatPencil/blob/master/outputs/demo.jpg?raw=true)
<small class="img-hint">Colorful</small>
 
As could be seen above, the grayscale image preserves the contour of the image to a large extent while at the same time giving different parts of the image distinctive levels of strength of pencil shading; the colorful image applies corresponding colors to the image, based on the grayscale.

I do have more examples, but readers are strongly encouraged to try cases of their own by running the program I provide in my GitHub repo [MatPencil](https://github.com/WellyZhang/MatPencil). **Play with it and have FUN!**

As shown above, the algorithm is also divided into two stages, one that generates line drawing with strokes and another that applies tone mapping and convolves the line drawing and tone mapping to produce the colorful image.

The first stage consists of four steps. In the first step, the program performs the edge detection, which in the paper is implemented by the second-order gradient. However, this is not fixed and other edge detection algorithms, such as Canny operator<sup>[[3]](#ref3)</sup> and Sobel operator<sup>[[4]](#ref4)</sup>, could be used. Then comes the convolution: the edge graph is convolved with eight equally distributed line kernels around a circle. After Non-Maximum-Suppression(NMS)-aware classification (no matter how I'm opposed to calling it *classification*), eight line kernels are again convolved with the eight classification responses and summed to yield the line drawing, *S'*. Proper postproessing would give us the grayscale image *S*.

In the second stage, the author combines the histogram tone mapping and the rendering. The author gathers many color distributions from colorful sketch images and summarizes the pattern by predefined distribution functions. A histogram matching is thus applied accordingly. However, to better render the textures, efforts are needed. In the paper, a loss function is desigend to work out an intermediate kernel *T*. The kernel again convolves with *S* and finally the colorful image comes out. 

This is just a brief overview of the steps of the algorithm and much more details are provided in the paper<sup>[[2]](#ref2)</sup>.

## Neural Art Style

Ever since the German computer scientists published their work<sup>[[5]](#ref5)</sup> on *arXiv*, enthusiasts around the world have been hurrying to implement the algorithm in their own languages and generate lots and lots of customized pictures. 

A famous [implementation](https://github.com/jcjohnson/neural-style) dates to Justin Johnson from Stanford by Torch. But I also prefer to present my own implementation here by Python, the [PyNeuralArt](https://github.com/WellyZhang/PyNeuralArt).

After setting the style image of *Starry Night* 

![Starry-Night](https://github.com/WellyZhang/PyNeuralArt/blob/master/images/starry_night.jpg?raw=true) 
<small class="img-hint">Starry Night</small>

and the content image a photo of Johannesburg,
 
![Johannesburg](https://github.com/WellyZhang/PyNeuralArt/blob/master/images/johannesburg.jpg?raw=true)
<small class="img-hint">Johannesburg</small>

I received a return exactly as expected, a ***starry Johannesburg***, by the computer rather than Vincent van Gogh.

![Starry-Johannesburg](https://github.com/WellyZhang/PyNeuralArt/blob/master/images/starry_johannesburg.jpg?raw=true)
<small class="img-hint">Starry Johannesburg</small>

As it turns out, there are even much more interesting paintings combining classic art style and contemporary art, for which someone even built a gallery.

Many people might wonder: why is it called ***NEURAL***? 

Actually, this algorithm leverages the most advanced technology of *neural networks* in *deep learning*.

> In machine learning and cognitive science, an *artificial neural network* (**ANN**) is a network inspired by biological neural networks (the central nervous systems of animals, in particular the brain) which are used to estimate or approximate functions that can depend on a large number of inputs that are generally unknown.<sup>[[6]](#ref6)</sup>

The algorithm first extracts image feature maps of the style image and the content image from a *Convolutional Neural Network*<sup>[[7]](#ref7)</sup>, or **CNN**, pretrained for image classification. Then a white noise image is fed to the neural network and by a properly designed loss function fusing the style image features and content image features and an optimizer, such as *L-BFGS*<sup>[[8]](#ref8)</sup>, a somewhat optimum solution would be found. If you are lucky enough setting the hyperparameters and training the network, such a solution would be an excellent combination of the style and the content.

For those of you unfamiliar with CNN and deep learning, [CS231n](http://cs231n.stanford.edu/) from Stanford would be a great start point.

## Deep Dream

Before diving into the basic idea of the Deep Dream algorithm, I'd love to demonstrate here one example to give you guys a feel of what it is.

![Input](https://github.com/WellyZhang/PyDream/blob/master/examples/input.jpg?raw=true)
<small class="img-hint">Original</small>
![Output](https://github.com/WellyZhang/PyDream/blob/master/examples/output.jpg?raw=true)
<small class="img-hint">Dream</small>

Actually, the deep dream algorithm is also closely related to the neural network, but unlike Neural Art Style, it does not condition the nonlinear image transformation on a style image. As an intern at Google first came up with this idea, he took a state-of-the-art image classification network, *GoogLeNet*, fed it with an image and as folks normally did in training neural networks, used the backpropagation algorithm<sup>[[9]](#ref9)</sup> to *update* the image. Gradually, an abstract painting emerged.

Nowadays, people have develeoped a couple of novel deep dream variants and even built galleries to show their results (google it to find a lot).

For those of you who want to try it yourselves, either access the original DeepDream repo from Google or [PyDream](https://github.com/WellyZhang/PyDream) from me. 

**Just get hands dirty on it!**

## Conclusion

From my point of view, the recently introduced Deep Neural Network Algorithms make prosperous the computational art, though as a side effect, but it is still plausible that naughty computer scientists would keep exploring this interdisciplinary field, well, as an ultimate way to relax after being driven crazy tuning the hyperparameters of a neural network.

## References

1. <a id="ref1">[Computational Art on CS145, Willamette University](https://www.willamette.edu/~gorr/classes/cs145Fa10/AlgArtWeb/index.htm)</a>
2. <a id="ref2">Cewu Lu, Li Xu, Jiaya Jia. Combining Sketch and Tone for Pencil Drawing Production. In *Proceedings of the Symposium on Non-Photorealistic Animation and Rendering*. Eurographics Association, 2012: 65-73.</a>
3. <a id="ref3">[Canny edge detector, Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector)</a>
4. <a id="ref4">[Sobel operator, Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator)</a>
5. <a id="ref5">Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. A Neural Algorithm of Artistic Style. *arXiv*: 1508.06576.</a>
6. <a id="ref6">[Artificial neural network, Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)</a>
7. <a id="ref7">[Convolutional neural network, Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)</a>
8. <a id="ref8">[Limited-memory BFGS, Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS)</a>
9. <a id="ref9">[Backpropagation, Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)</a>
