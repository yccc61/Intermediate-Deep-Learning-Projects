"""
Fall 2024, 10-417/617
Homework 2
Programming: CNN
  
IMPORTANT:
    DO NOT change any function signatures

Sep 2024
"""

import numpy as np
import im2col_helper  # uncomment this line if you wish to make use of the im2col_helper.pyc file for experiments

def im2col(X, k_height, k_width, padding=1, stride=1):
    """
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    
    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    """
    #padding x
    pad_width=((0,0),(0,0),(padding,padding),(padding, padding))
    X_padded=np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
    X_shape=np.shape(X) #0:N, 1:C, 2:H, 3:W
    output_H=(X_shape[2]-k_height+2*padding)//stride+1
    output_W=(X_shape[3]-k_width+2*padding)//stride+1
    counter=0
    result=np.zeros((X_shape[1]*k_height*k_width,output_H*output_W*X_shape[0]))
    for h in range (output_H):
        for w in range (output_W):
            for n in range (X_shape[0]):
                curr_col=X_padded[n,:, (h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)]
                reshapped_curr_col=curr_col.reshape(-1,)
                result[:,counter]=reshapped_curr_col
                counter+=1
    return result





def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    """
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    """
    (orig_N, orig_C, orig_H, orig_W)=X_shape
    output_H=(orig_H-k_height+2*padding)//stride+1
    output_W=(orig_W-k_width+2*padding)//stride+1
    result=np.zeros((orig_N, orig_C,orig_H+2*padding, orig_W+2*padding))
    counter=0
    for h in range (output_H):
        for w in range (output_W):
            for n in range (orig_N):
                curr_conv=grad_X_col[:,counter].reshape(orig_C,k_height, k_width)
                result[n,:,(h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)]+=curr_conv
                counter+=1
    res_H=result.shape[2]
    res_W=result.shape[3]
    return result[:,:,padding:res_H-padding, padding:res_W-padding]
        



class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """

    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class LeakyReLU(Transform):
    """
    Implement this class
    """
    def __init__(self, alpha, dropout_probability=0.5):
        
        Transform.__init__(self)
        self.alpha=alpha
        self.dropout_probability=dropout_probability
        self.dropout_mask=None

    def forward(self, x, train=False):
        """
        :param x: input matrix
        :param train: optional param indicating training

        :return: output matrix

        NOTE: (IMPORTANT!!!)
        Call np.random.uniform(0, 1, x.shape) exactly once in this function.
        The behavior of this function should change wrt the train param.

        Hint: you may find np.where useful for this.
        """
        self.orginal_x=x
        self.leaky=np.where(x<0, self.alpha*x, x)
        if train==True:
            prob_drop=np.random.uniform(0,1, x.shape)
            self.dropout_mask=np.where(prob_drop<self.dropout_probability, 0, 1)
            self.leaky=self.leaky*self.dropout_mask
        return self.leaky


    def backward(self, grad_wrt_out):
        """
        :param grad_wrt_out: gradient matrix from previous Transform
        """
        grad_leaky=np.where(self.orginal_x<=0, self.alpha, 1)
        if self.dropout_mask is not None:
            grad_leaky=grad_leaky*self.dropout_mask

        return grad_leaky*grad_wrt_out

class Flatten(Transform):
    """
    Implement this class
    """

    def forward(self, x):
        """
        returns Flatten(x)
        """
        num_filters=np.shape(x)[0]
        self.shape=np.shape(x)
        return x.reshape(num_filters, -1)

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        return dloss.reshape(self.shape)


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """

    def __init__(self, input_shape, filter_shape, rand_seed=None):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        self.C, self.H, self.Width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        
        b=np.sqrt(6.0/((self.num_filters+self.C)*self.k_height*self.k_width))
        self.W=np.random.uniform(-b, b, (self.num_filters, self.C, self.k_height, self.k_width))
        self.flatten=Flatten()
        self.b=np.zeros((self.num_filters,1))


        self.momen_W=np.zeros_like(self.W)
        self.momen_b=np.zeros_like(self.b)

        self.grad_W=np.zeros_like(self.W)
        self.grad_b=np.zeros_like(self.b)

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here to vectorize your computations
        """
        self.stride=stride
        self.pad=pad
        self.inputs=inputs
        self.col_X=im2col_helper.im2col(inputs, self.k_height, self.k_width, pad, stride)
        self.flatten_W=self.flatten.forward(self.W)
        self.forward_res=self.flatten_W@self.col_X+self.b

        (batch_size, _, height, width)=np.shape(inputs)
        output_H=(height-self.k_height+2*pad)//stride+1
        output_W=(width-self.k_width+2*pad)//stride+1
        self.forward_res = self.forward_res.reshape(self.num_filters, output_H, output_W, batch_size)
        self.forward_res = self.forward_res.transpose(3, 0, 1, 2)
        return self.forward_res


    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        use im2col_bw here to vectorize your computations
        """
        reshaped_dout=(dloss.transpose(1,2,3,0)).reshape(self.W.shape[0], -1)
        #compute the gradient for layer
        #rearrange dloss so num of filters come first
        self.grad_layers=self.flatten_W.T@reshaped_dout
        self.grad_layers=im2col_helper.im2col_bw(self.grad_layers,np.shape(self.inputs), self.k_height, self.k_width, self.pad, self.stride)
        # compute the gradient for weights
        self.grad_W = reshaped_dout @ self.col_X.T
        self.grad_W= self.grad_W.reshape(np.shape(self.W))
        #compute the gradient for b
        self.grad_b=np.sum(reshaped_dout,axis=1,keepdims=True)
        return [self.grad_W, self.grad_b, self.grad_layers]

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as in HW1 MLP
        """
        self.momen_W=momentum_coeff*self.momen_W+self.grad_W
        self.momen_b=momentum_coeff*self.momen_b+self.grad_b
        self.W=self.W-learning_rate*self.momen_W
        self.b=self.b-learning_rate*self.momen_b

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """

    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_height, self.filter_width=filter_shape
        self.stride=stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        """
        (N, C, H, W)=np.shape(inputs)
        out_H=(H-self.filter_height)//self.stride+1
        out_W=(W-self.filter_width)//self.stride+1
        self.inputs=inputs
        self.forward_res=np.zeros((N,C,out_H, out_W))
        for h in range (out_H):
            for w in range (out_W):
                target=inputs[:,:,self.stride*h: self.stride*h+self.filter_height, self.stride*w:self.stride*w+self.filter_width]
                max_target=np.max(target, axis=(2, 3), keepdims=True)
                self.forward_res[:,:,h,w]=max_target.reshape(2,1)
        return self.forward_res
        

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        # expand dloss in forward_input shape, where element are zero if not max
        (N, C, H, W)=np.shape(dloss)
        backward_res=np.zeros_like(self.inputs) 
        for h in range (H):
            for w in range(W):
                for n in range (N):
                    for c in range (C):
                        target=self.inputs[n,c,h*self.stride:(h*self.stride+self.filter_height),w*self.stride:(w*self.stride+self.filter_width)]
                        max_index = np.unravel_index(np.argmax(target), (self.filter_height, self.filter_width))
                        backward_region=backward_res[n,c,h*self.stride:(h*self.stride+self.filter_height),w*self.stride:(w*self.stride+self.filter_width)]
                        backward_region[max_index] = dloss[n, c, h, w]
                
        return backward_res
        
        
def random_weight_init(input, output):
    """
    Initializes random weight vector

    :param input: input dimension
    :param output: output dimension
    :return: (output x input) matrix with random weights
    """
    b = np.sqrt(6) / np.sqrt(input + output)
    return np.random.uniform(-b, b, (input, output))

def zeros_bias_init(outd):
    """
    Initializes zero bias vector

    :param output: output dimension
    :return: (output x 1) matrix with zeros
    """
    return np.zeros((outd, 1))

class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """

    def __init__(self, indim, outdim, rand_seed=None):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        
        self.indim=indim
        self.outdim=outdim
        #weights_size(outdim, indim)
        self.W=random_weight_init(indim, outdim)
        self.b=zeros_bias_init(outdim)

        self.grad_weights=np.zeros((indim, outdim))
        self.grad_b=np.zeros((outdim,1))

        self.momen_weights=np.zeros((indim, outdim))
        self.momen_b=np.zeros((outdim,1))

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.inputs=inputs
        return inputs@self.W+self.b.T

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        self.grad_b=np.sum(dloss, axis=0, keepdims=True).T
        self.grad_weights=(self.inputs).T @ dloss
        self.grad_inputs = dloss@self.W.T
        return [self.grad_weights, self.grad_b, self.grad_inputs]

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.momen_weights=momentum_coeff*self.momen_weights+self.grad_weights
        self.momen_b=momentum_coeff*self.momen_b+self.grad_b
        self.W=self.W-learning_rate*self.momen_weights
        self.b=self.b-learning_rate*self.momen_b

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class SoftMaxCrossEntropyLoss:
    """
    Implement this class
    """

    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be the mean loss over the batch)
        """
        pass

    def backward(self):
        """
        return shape (batch_size, num_classes)
        Remeber to divide by batch_size so the gradients correspond to the mean loss
        """
        pass

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        pass


class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> LeakyRelu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None,leakyReluAlpha=0.1):
        """
        Initialize Conv, LeakyReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SoftMaxCrossEntropy object.
        Remember to pass in the rand_seed to initialize all layers,
        otherwise you may not pass autograder.
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> LeakyRelu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None, leakyReluAlpha=0.1):
        """
        Initialize Conv, LeakyReLU, MaxPool, Conv, LeakyReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape (batch, channels, height, width)
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> LeakyRelu -> MaxPool -> Conv -> LeakyRelu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None, leakyReluAlpha=0.1):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 1x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass



class ConvNetFour:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> LeakyRelu -> MaxPool -> Conv -> LeakyRelu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None, leakyReluAlpha=0.1):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 5x4x4
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass
    

class ConvNetFive:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> LeakyRelu -> MaxPool -> Conv -> LeakyRelu -> Conv -> LeakyRelu -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None, leakyReluAlpha=0.1):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply LeakyRelu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 7x3x3
        then apply LeakyRelu
        then Conv with filter size of 7x3x3
        then apply LeakyRelu
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass

def labels2onehot(labels):
    return np.eye(np.max(labels) + 1)[labels].astype(np.float32)


if __name__ == "__main__":
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    import pickle

    # change this to where you downloaded the file,
    # usually ends with 'cifar10-subset.pkl'
    CIFAR_FILENAME = "../cifar10-subset.pkl"
    with open(CIFAR_FILENAME, "rb") as f:
        data = pickle.load(f)

    # preprocess
    trainX = data["trainX"].reshape(-1, 3, 32, 32) / 255.0
    trainy = labels2onehot(data["trainy"])
    testX = data["testX"].reshape(-1, 3, 32, 32) / 255.0
    testy = labels2onehot(data["testy"])
