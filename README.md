# DCGAN_Implementation

# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are models capable of generating specific outputs after being fed with a random noise vector. They are trained via an `adversarial process` formulated by Ian Goodfellow in the paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). They are composed by a generative network that creates a distribution similar to the training distribution, and a discriminative network that tries to distinguish between real training examples and fake training examples. The adversarial process is an "implicit" method, as the network improves its performances by generating more "realistic" inputs. The training involves two "players": a Generator and a Discriminator. The Generator receives as input a simple distribution, and produces an output that is the sampling from that input distribution. A subsequent Discriminator will judge if the sample distribution is similar to the real data distribution. If not the generator will produce a second output that tries to reduce the difference with the real data.



<img width="954" alt="Screenshot 2022-08-30 at 12 58 46" src="https://user-images.githubusercontent.com/80494835/193601253-451f0709-ffed-4bd5-8773-650dd708b7b2.png">

The goal is to train the Generator and the Discriminator jointly in a adversarial training:

$$
    \min_{\theta_g} \max_{\theta_d} [E_{x \sim p_{data}} \log D_{\theta_d}(x)+E_{z \sim p_{z}}\log (1-D_{\theta_d}(G_{\theta_d}(z)))]
$$

The term $D_{\theta_d}(x)$ represents the discriminator output for real data, while $D_{\theta_d}(G_{\theta_d}(z)))$ the discriminator output for fake data $G(z)$s. The discriminator $\theta_d$ wants to maximise the objective function such that $D_{\theta_d}(x)$ is close to 1 (real) and $D_{\theta_d}(G_{\theta_d}(z)))$ is close to 0 (fake). At the opposite, the generator $G_{\theta_g}$ wants to minimise the objective function such that $D_{\theta_d}(G_{\theta_d}(z)))$ is close to 1.


## Implementation strategy: Original GAN training algorithm

To train a GAN we should follow this steps, taken from the original 2014 paper by Goodfellow, et al. "[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)".


**Algorithm 1**



**for** number of training iterations **do**

> **for** k steps **do**
  
>> Sample m noise vectors
    
>> Sample m real images
    
>> compute the gradient $\nabla_{\theta_d}$ of 
>> $$\frac{1}{m} \sum_{i=1}^m [ \log D(x^{(i)}) + \log (1 - D(G(z^{(i)}))) ]$$

>> update the discriminator with SGD by ascending it (even if in DCGANs it will be minimised)

> end **for**

> Sample m noise vectors
> compute the gradient $\nabla_{\theta_g}$ of 
> $$\frac{1}{m} \sum_{i=1}^m [\log (1 - D(G(z^{(i)}))) ]$$

> update the generator with SGD by descending it

end **for**


We can observe that the training loop uses two version of th ebinary cross entropy and maximise the cost function to update the discriminator parameters, while it minimise the cost function when updating the generator's parameters.
     
## Different types of GANs

https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c

# DCGANs

DCGANs are composed by a pair of:

- Generator: the goal of the generator is to output an image of the same size of the training images (in this case 3x64x64). "This is accomplished through a series of strided two dimensional convolutional transpose layers", each paired with a 2d batch norm layer and a relu activation.


- Discriminator: "is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). Here, DD takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both GG and DD."

"We pass random seeds into the generator and it outputs images. These images are passed to the Discriminator during new rounds of traning as the fake images. When training the Discriminator we will pass in images from the traning set (real) and images from the generator (fake) and the role of the discriminator will be to correctly and confidently differentiate between the real and fake images"_ [cite1](https://nabeelvalley.co.za/docs/data-science-with-python/image-classification-with-keras/)
 
# Implementing DCGAN:

A DCGAN was first described by Radford et. al. in the paper [Unsupervised representation learning with deep convolutional generative adversarial networks, A. Radfordet al, arXiv, 2015](https://arxiv.org/abs/1511.06434) and are a class of CNNs identified after an extensive model exploration. Deep convolutional generative adversarial networks (DCGANs) are made by an adversarial pair that contains a discrimator and a generator network, both using convolutional layers and some important architectural constrains are introduced [1] to make this networks able to learn in a consistent way:

-  Deterministic spatial pooling functions (such as maxpooling) are replaced with `strided convolutions` (when the stride is larger than one, one usually talks about strided convolution to make the difference explicit) making the generator able to learn its own downsapling.

- Eliminating fully connected layers on top of convolutional features.

- Doing Batch Normalization which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance.

- Use ReLU activation in generator for all layers except for the output, which uses Tanh.

- Use `LeakyReLU activation` in the discriminator for all layers (Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope, determined before training, for negative values instead of a flat slope and it is popular using it in tasks where we we may suffer from sparse gradients, such as training generative adversarial network).

In the paper, the authors also give some tips about how to setup the optimizers, how to calculate the loss functions, and how to initialize the model weights [1]:

- No pre-processing is applied to training images besides scaling to the range of the tanh activation function [-1, 1]

- Training is done with stochastic gradient descent (SGD) with a mini-batch size of 128

- All weights are initialized from a zero-centered Normal distribution with standard deviation 0.02

- In the LeakyReLU, the slope of the leak was set to 0.2 in all models

- The optimizer used is Adam

- Learning rate used is 0.0002

- The momentum term β1 is reduced from 0.9 to 0.5 to help stabilizing the training.

Implementing a GAN is more an art than a science and [here](https://github.com/soumith/ganhacks) some more guidelines on how to train a GAN.

## Walkthrough:

Implementing the DCGAN will require to follow this steps:

- Import Libraries 
- Initialise hyperparameters
- Load and Preprocess the Data
- Create the dataloader
- Initialise weights

- **Build Generator's networks**
- **Build Discriminator's network**
- **Define the Loss function and the Optimizer**
- **Define training loop**

- Training the network

- Results: Generating Synthetic Images

Let us now take a look at the details of the steps that are the most relevant.

## Generator

The generator's task is to accept a one dimensional noise vector (z) of size 100 and converting it into a 3x64x64. It is made of a series of `2D strided convolutional-transpose layers` (The transposed Convolutional Layer is also wrongfully known as the Deconvolutional layer. A deconvolutional layer reverses the operation of a standard convolutional layer. Transposed convolution doesn’t reverse the standard convolution by values, rather by dimensions only), batch norm layers, and ReLU activations. The input is a latent vector, z, that is drawn from a standard normal distribution and the output is a 3x64x64 RGB image. The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image [2]. The output is given after a Tanh activation function. 

```python
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

## Discriminator

The discriminator is made up of `strided convolution layers`, batch norm layers, and `LeakyReLU activations`. The input is a 3x64x64 input image and the output is a scalar probability that the input is from the real data distribution [2], and the output is given after a  Sigmoid activation function. 

```python
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
```

## Training

We will use a Binar Cross Entropy (BCE) as loss function for training our network, and we will use as a convetion the following labels:

- True = 1 (if the ground truth is a real image)
- False = 0 (if the ground truth is a fake image)

The training of the DCGAN will closely follow Algorithm 1 from Goodfellow’s paper, while abiding by some of the best practices shown in [ganhacks](https://github.com/soumith/ganhacks).

### Training the Discriminator

For training the Discriminator we will follow the same logic as the original GAN, by wanting to maximise the cost of $$\log D(x) + \log (1- D(G(z)))$$ where $D(x)$ is the prediction of the discriminator given a real image $x$, and $D(G(z))$ is the prediction of the discriminator given a fake image $G(z)$, generated from a vector of random noise $z$.

We can do this in two steps:

1- create a batch of real images to forward pass in the discriminator and compute the gradients with a backward pass

2- create a batch of fake images with G to forward pass in D and accumulate the gradients with a backward pass

Finally we can pass the gradients to the optimizer to calculate the updating step.

### Training the Generator

Training the Generator in the original algorithm is obtained by minimising the cost of $$\log(1-D(G(z)))$$ while in the DCGAN paper we will instead maximise the cost of $$\log (D(G(z)))$$ using real labels as the ground truth. The update of the weights will happen after that the batch of fake images generated in the previous step are classified by the discriminator, the loss of G is computed, and the gradient of G is calculated with a backward pass.

Please not that even if using the real labels as the ground truth might seems counter intuitive, it allows us to use the $\log(x)$ part of the BCE instead of the $\log(1-x)$ part.

# Challenges

## Building a model in PyTorch

PyTorch offer a very flexible framework to build networks with. Building neural networks involves building a class where to define the `layers` of the model and the `forward()` method, that defines the way that the input is passed from each layer from the input to the output stage. PyTorch offers many different layers, such as Linear for fully connected layers, Conv2d for convolutional layers, Recurrent Layers for Recurrent neural networks (or RNNs), and other layers such as Normalization layers , Dropout layers and MaxPooling layers. Also many Activation functions are available such as ReLU, Softmax, and Sigmoid.

Please refer to this notebook to understand more deeply what building a model in PyTorch involves: 

https://github.com/draperkm/Frameworks_notes/blob/main/Building_Models.ipynb


## Visualizing Tensors as RGB Images

When working with PyTorch, we are in reality dealing with Tensors (Arrays) data structures, and treating them is not always as straightforward as it might seems. You can visit the following link to see how I treated this problem: 

https://github.com/draperkm/Frameworks_notes/blob/main/Visualising_Tensors_PyTorch.ipynb


# Experiments and results

We have trained this DCGAN network for 10 - 30 - 60 epochs. The results that we can obtain from this implementation have plateaud after 30 epochs, as we can not notice significative differences between the two final results. 

## The original data set

The data set has been retrieved on Kaggle: https://www.kaggle.com/datasets/gasgallo/faces-data. Here some examples extracted from the data set:

<p align="center">
<img width="700" alt="Screenshot 2022-10-03 at 18 50 17" src="https://user-images.githubusercontent.com/80494835/193644756-e50e2195-4e86-40a7-92be-6d2cc053ecc6.png">
</p>

## Results


The program created next will generate faces similar to these:

<p align="center">
<img width="1231" alt="Screenshot 2022-10-03 at 18 34 59" src="https://user-images.githubusercontent.com/80494835/193643851-268c4b61-9c68-4ea6-8aa1-498a4fef87e7.png">
</p>

It is evident that this type of implementation can allow us to reach only a certain level of quality in the generated images, but we demonstrated how it is possible to build and train a GAN on a simple laptop Machine with the help of Google Colab. To produce images of a higher quality (and resolution) we will have to take a look at different GAN implementation such a `StyleGAN2`.

<img width="1140" alt="Screenshot 2022-10-03 at 18 40 42" src="https://user-images.githubusercontent.com/80494835/193644187-77f33e20-7cc8-4d41-bf00-8d3f3c445aaf.png">



# References


1- https://arxiv.org/abs/1511.06434 DCGAN


2- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

3- t81_558_class_07_2_Keras_gan.ipynb, Jeff Heaton

4- https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11

5- https://arxiv.org/pdf/1406.2661.pdf GANs

6- https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

7- https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html


