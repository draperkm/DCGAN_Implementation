# DCGAN_Implementation

# Generative Adversarial Networks (GANs)

These netwroks are composed by a pair of:

- Generator: the goal of the generator is to output an image of the same size of the training images (in this case 3x64x64). "This is accomplished through a series of strided two dimensional convolutional transpose layers", each paired with a 2d batch norm layer and a relu activation.


- Discriminator: "is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). Here, DD takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both GG and DD."


![gans1](https://user-images.githubusercontent.com/80494835/189524622-1c975c2c-7e13-4be4-8944-edf9f334a030.jpg)

"We pass random seeds into the generator and it outputs images. These images are passed to the Discriminator during new rounds of traning as the fake images. When training the Discriminator we will pass in images from the traning set (real) and images from the generator (fake) and the role of the discriminator will be to correctly and confidently differentiate between the real and fake images"_ [cite1](https://nabeelvalley.co.za/docs/data-science-with-python/image-classification-with-keras/)

## Cost function in the original GAN

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h(x^{(i)}, \theta) + (1-y^{(i)}) \log (1 - h(x^{(i)}, \theta)) ] $$

where $J(\theta)$ is the avarage cost for a given set of parameters, m is the batch size, $y^{(i)}$ is the label of example i and $h(x^{(i)}, \theta)$ is the prediction given by the model for the example $x^{(i)}$. In this case the prediction is made from the discriminator???.
So this this function can be seen as two main terms: the first one is a function that when the prediction is close to the real one the loss is close to zero, while when the prediction is far from the true lable the loss approaches infinity. The second term does the opposite thing. Source: https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/lecture/2bF5q/bce-cost-function

## Original GAN training algorithm

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
     

## Types of different GANs and DCGANs

https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c
 
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

Training the Generator in the original algorithm is obtained by minimising the cost of $ \log(1-D(G(z)))$ while in the DCGAN paper we will instead maximise the cost of $$\log (D(G(z)))$$ using real labels as the ground truth. The update of the weights will happen after that the batch of fake images generated in the previous step are classified by the discriminator, the loss of G is computed, and the gradient of G is calculated with a backward pass.

Please not that even if using the real labels as the ground truth might seems counter intuitive, it allows us to use the $\log(x)$ part of the BCE instead of the $\log(1-x)$ part.

# Challenges

## DataLoader num_workers - Generating data in parallel with PyTorch

1- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

2-https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/7

"Are you sure that memory usage is the most serious overhead ? What about IO usage ?
Setting too many workers might cause seriously high IO usage which can become very uneffective.

I would love to get your advice about the recommended way to deal with my data - I feed my CNN with large batches (256/512/1024…) of small patches of size 50x50. I intend to use the ImageFolder DataLoader for that, but I’m afraid that it would be very uneffective to load from disk a lot of small images in high frequency."

"if the data set is small like cifar10, why doesn’t the whole data set stay in the GPU the whole time? Why would # workers do anything?"

"The more data you put into the GPU memory, the less memory is available for the model.
If your model and data is small, it shouldn’t be a problem. Otherwise I would rather use the DataLoader to load and push the samples onto the GPU than to make my model smaller."

## Choosing Mini-Batches 

https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network

"The question has been asked a while ago but I think people are still tumbling across it. For me it helped to know about the mathematical background to understand batching and where the advantages/disadvantages mentioned in itdxer's answer come from. So please take this as a complementary explanation to the accepted answer.

Consider Gradient Descent as an optimization algorithm to minimize your Loss function 𝐽(𝜃). The updating step in Gradient Descent is given by
$$\theta_{k+1} = \theta_{k} - \alpha \nabla J(\theta)$$

"

## Building a model in PyTorch

Defining a network in PyTorch might be a challenge. Please refer to this notebook to understand more deeply what building a model in PyTorch involves: 

(https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)

https://github.com/draperkm/Frameworks_notes/blob/main/Building_Models.ipynb


## Preprocessing data: Printing Tensors as an image

Printing RGB images requires tensors to be clipped between 0 and 1 if float values or betweeen 0 and 255 if integers.

https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html


# Experiments and results

"The program created next will generate faces similar to these. While these faces are not perfect, they demonstrate how we can construct and train a GAN on or own. Later we will see how to import very advanced weights from nVidia to produce high resolution, realistic looking faces." (Heaton)

## Making DCGAN better: StyleGAN2

Other apporaches can be adopted to obtain better generated images. High resolution...

# References

1- https://arxiv.org/abs/1511.06434 DCGAN

2- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

3- t81_558_class_07_2_Keras_gan.ipynb, Jeff Heaton

4- https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11

5- https://arxiv.org/pdf/1406.2661.pdf GANs

6- https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

7- https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html


