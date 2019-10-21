# Cheat sheet
## Layers
### Dense
Fully connected layer. Each neuron connected to each neuron in the previous layer. Most commonly used layer type.
### Flatten
Creating a 1D vector from a 2D matrix. Commonly used to feed an image to a dense layer.
## Activation functions
### Linear
$g(z)=z$

Only used in output layer, in the case of regression when the labels are not bounded.
### Sigmoid
$g(z)=\frac{1}{1+e^{-z}}$

Transforms parameters between 0 and 1. Used in output layer if the label is a probability.
### TanH
$g(z)=\frac{e^x-e^{-z}}{e^{-z}+e^z}$

Transforms parameters between -1 and 1. Used for bounding.
### ReLu
$g(z)=\text{max}(0,z)$

Most common in hidden layers.
### Softmax
$g(z_i)=\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}}$

Only used in output layer, in the case of categorisation. Transforms the layer into a probability density function.
## Loss functions
### MSE
$L(y,\hat{y})=\frac{(y-\hat{y})^2}{2}$

Used for regression.
### MAE
$L(y,\hat{y})=\lvert y-\hat{y} \rvert$

Used for regression.
### Categorical crosentropy
$L(y,\hat{y})=-\sum_{c=1}^C{y \ln{\hat{y}}}$

Used for classification, if the labels are one-hot encoded. One-hot encoding: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
### Sparse categorical crosentropy
$L(y,\hat{y})=-\sum_{c=1}^C{y \ln{\hat{y}}}$

Used for classification, if the labels are not one-hot encoded. Not one-hot encoding: [4]
