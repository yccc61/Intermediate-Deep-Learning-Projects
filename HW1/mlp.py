"""
Fall 2024, 10-417/617
Assignment-1

IMPORTANT:
    DO NOT change any function signatures

September 2024
"""
import numpy as np
import matplotlib.pyplot as plt
def random_weight_init(input, output):
    """
    Initializes random weight vector

    :param input: input dimension
    :param output: output dimension
    :return: (output x input) matrix with random weights
    """
    b = np.sqrt(6) / np.sqrt(input + output)
    return np.random.uniform(-b, b, (output, input))

def zeros_bias_init(outd):
    """
    Initializes zero bias vector

    :param output: output dimension
    :return: (output x 1) matrix with zeros
    """
    return np.zeros((outd, 1))

def labels2onehot(labels):
    """
    Returns one-hot representation of labels

    :param labels: list/vector of labels
    :return: (len(labels) x 12) one-hot matrix
    """
    return np.array([[i==lab for i in range(12)] for lab in labels])


class Transform:
    """
    This is the base class. You do not need to change anything.
    All functions to be implemented are marked with TODO.

    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters (weights, biases, etc.)
        """
        pass

    def forward(self, x):
        """
        Function for forward pass

        :param x: input matrix (passed as column vectors)
        :return: Transform(x)
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Function for backward pass (does NOT apply gradient)

        :param grad_wrt_out:
            gradient matrix from next Transform (i.e. next layer in forward pass)
        :return: grad_wrt_x (which will be grad_wrt_out for previous Transform)

        NOTE:
        In this function, we accumulate and save the gradient values instead
        of assigning the gradient values.

        This allows us to call forward and backward multiple times while
        only updating the parameters once.

        We will apply the gradients in step().
        """
        pass

    def step(self):
        """
        Function for applying gradient accumulated from backward()

        NOTE:
        This function should update the parameters initialized in __init__()
        """
        pass

    def zerograd(self):
        """
        Function for reseting the graduents. Usually called before backward().
        """
        pass


class LeakyReLU(Transform):
    """
    Implement this class

    LeakyReLU non-linearity, combined with dropout
    IMPORTANT the Autograder assumes these function signatures
    """
    def __init__(self, alpha, dropout_probability=0):
        """
        :param dropout_probability: dropout probability
        """
        self.alpha=alpha
        self.dropout_probability=dropout_probability

        self.dropout_mask=None

    def forward(self, x, train=True):
        """
        :param x: (outdim, batch_size) input matrix
        :param train: optional param indicating training

        :return: (outdim, batch_size) output matrix

        NOTE: (IMPORTANT!!!)
        Call np.random.uniform(0, 1, x.shape) exactly once in this function.
        The behavior of this function should change wrt the train param.

        Hint: you may find np.where useful for this.
        """
        self.x=x
        self.leaky=np.where(x<0, self.alpha*x, x)
        if train==True:
            prob_drop=np.random.uniform(0,1, x.shape)
            self.dropout_mask=np.where(prob_drop<self.dropout_probability, 0, 1)
            self.leaky=self.leaky*self.dropout_mask
        else:
            expectation=1-self.dropout_probability
            self.leaky=expectation*self.leaky

        return self.leaky


    def backward(self, grad_wrt_out):
        """
        :param grad_wrt_out:
            (outdim, batch_size) gradient matrix from previous Transform

        Hint: you may find np.where useful for this
        """
        grad_leaky=np.where(self.x<=0, self.alpha, 1)
        if self.dropout_mask is not None:
            grad_leaky=grad_leaky*self.dropout_mask

        return grad_leaky*grad_wrt_out


class LinearMap(Transform):
    """
    Implement this class

    NOTE:
    For consistency, please use random_weight_init() and zero_bias_init()
    given at the top for initialization
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        """
        :param indim: input dimension
        :param outdim: output dimension
        :param alpha: momentum update param
        :param lr: learning rate
        """
        Transform.__init__(self)
        self.indim=indim
        self.outdim=outdim
        self.alpha=alpha
        self.lr=lr
        #weights_size(outdim, indim)
        self.weights=random_weight_init(indim, outdim)
        self.b=zeros_bias_init(outdim)

        self.grad_weights=np.zeros((outdim, indim))
        self.grad_b=np.zeros((outdim,1))

        self.momen_weights=np.zeros((outdim, indim))
        self.momen_b=np.zeros((outdim, 1))



    def forward(self, x):
        """
        :param x: (indim, batch_size) input matrix
        :return: (outdim, batch_size) output matrix
        """
        self.x=x
        return self.weights@x+self.b


    def backward(self, grad_wrt_out):
        """
        :param grad_wrt_out:
            (outdim, batch_size) gradient matrix from next Transform
        :return grad_wrt_x:
            (indim, batch_size) gradient matrix for previous Transform

        NOTE:
        Your backward call should accumulate gradients.
        y=w^T x+b
        dL/dw^T=dl/dy* (dy/d^W^T)->x
        """
        self.grad_weights+=grad_wrt_out@(self.x).T
        #keepdims=True so that it is still a 2d matrix with 1 colum
        self.grad_b+=np.sum(grad_wrt_out, axis=1, keepdims=True)
        return (self.weights.T)@grad_wrt_out

    def step(self):
        """
        Apply gradients calculated by backward() to update the parameters

        NOTE:
        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        self.momen_weights=self.alpha*self.momen_weights+self.grad_weights
        self.momen_b=self.alpha*self.momen_b+self.grad_b
        self.weights=self.weights-self.lr*self.momen_weights
        self.b=self.b-self.lr*self.momen_b

    def zerograd(self):
        self.grad_weights=np.zeros((self.outdim, self.indim))
        self.grad_b=np.zeros((self.outdim,1))


    def getW(self):
        """
        :return: (outdim, indim), i.e. W shape
        """
        return self.weights

    def getb(self):
        """
        :return: (outdim, 1), i.e. b shape
        """
        return self.b

    def loadparams(self, w, b):
        """
        Load parameters of linear layer (for autograder)

        :param w: weight matrix
        :param b: bias matrix
        """
        self.weights=w
        self.b=b


class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        :param logits:
            (num_classes, batch_size) matrix of pre-softmax scores
        :param labels:
            (num_classes, batch_size) matrix of true labels of given inputs

        :return: loss as scalar
            (your loss should be the mean value over the batch)

        NOTE:
        This means both the logits and labels are one-hot encoded
        """
        exp_logits=np.exp(logits)
        denominator=np.sum(exp_logits, axis=0, keepdims=True)
        soft_max=exp_logits/denominator
        log_soft_max=np.log(soft_max)

        self.batch_size=np.shape(logits)[1]
        loss=-np.sum(log_soft_max*labels)/self.batch_size

        self.softmax=soft_max
        self.labels=labels
        return loss


    def backward(self):
        """
        :return: (num_classes, batch_size) gradient matrix

        NOTE:
        Don't forget to divide by batch_size because your loss is a mean
        """
        self.grad_logit=(self.softmax-self.labels)/self.batch_size
        return self.grad_logit

    def getAccu(self):
        """
        Return accuracy here (as you wish).
        This part is not autograded.
        """

        one_hot_softmax=(np.max(self.softmax, axis=0)==self.softmax).astype(int)

        num_corrects=np.sum(one_hot_softmax*self.labels)
        accuracy=num_corrects/np.shape(one_hot_softmax)[1]

        return accuracy




class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, outdim, hiddenlayer=100,
                 alpha=0.1, leakyReluAlpha=0.1,dropout_probability=0, lr=0.01):
        """
        :param indim: input dimension
        :param outdim: output dimension
        :param hiddenlayer: hidden layer dimension
        :param alpha: momentum update param
        :param dropout_probability: dropout probability
        :param lr: learning rate
        """
        Transform.__init__(self)
        self.linear=LinearMap(indim=indim, outdim=hiddenlayer, alpha=alpha, lr=lr)
        self.activation=LeakyReLU(alpha=leakyReluAlpha, dropout_probability=dropout_probability)
        self.hiddenLayer=LinearMap(indim=hiddenlayer, outdim=outdim, alpha=alpha, lr=lr)


    def forward(self, x, train=True):
        """
        :param x: (indim, batch_size) input matrix
        :param train: optional param indicating training
        """
        linear_layer=self.linear.forward(x)
        activation=self.activation.forward(linear_layer, train)
        result=self.hiddenLayer.forward(activation)
        return result

    def backward(self, grad_wrt_out):
        """
        :param grad_wrt_out:
            (outdim, batch_size) gradient matrix from next Transform
        :return grad_wrt_x:
            (indim, batch_size) gradient matrix for previous Transform
        """
        hidden_back=self.hiddenLayer.backward(grad_wrt_out)
        act_back=self.activation.backward(hidden_back)
        linear_back=self.linear.backward(act_back)
        return linear_back

    def step(self):
        self.linear.step()
        self.hiddenLayer.step()

    def zerograd(self):
        self.linear.zerograd()
        self.hiddenLayer.zerograd()

    def loadparams(self, Ws, bs):
        """
        Load parameters (for autograder)

        :param Ws: weights array list, first layer first
            e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        :param bs: biases array list, first layer first
            e.g., Ws may be [LinearMap1.b, LinearMap2.b]

        NOTE:
        Use LinearMap.loadparams() to implement this.
        """
        self.linear.loadparams(Ws[0], bs[0])
        self.hiddenLayer.loadparams(Ws[1],bs[1])

    def getWs(self):
        """
        Return the weights for each layer, according to description in loadparams()
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        """
        w_linear=self.linear.getW()
        w_hiddenlinear=self.hiddenLayer.getW()
        return [w_linear, w_hiddenlinear]

    def getbs(self):
        """
        Return the biases for each layer, according to description in loadparams()
        e.g., bs may be [LinearMap1.b, LinearMap2.b]
        """
        b_linear=self.linear.getb()
        b_hiddenlinear=self.hiddenLayer.getb()
        return  [b_linear, b_hiddenlinear]


class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, indim, outdim, hiddenlayers=[100,100],
                 alpha=0.1, leakyReluAlpha=0.1, dropout_probability=0, lr=0.01):
        """
        :param indim: input dimension
        :param outdim: output dimension
        :param hiddenlayers: hidden layers 1 and 2 dimensions
        :param alpha: momentum update param
        :param dropout_probability: dropout probability
        :param lr: learning rate
        """
        self.linear1=LinearMap(indim=indim, outdim=hiddenlayers[0], alpha=alpha, lr=lr)
        self.activation1=LeakyReLU(alpha=leakyReluAlpha, dropout_probability=dropout_probability)
        self.linear2=LinearMap(indim=hiddenlayers[0], outdim=hiddenlayers[1], alpha=alpha, lr=lr)
        self.activation2=LeakyReLU(alpha=leakyReluAlpha, dropout_probability=dropout_probability)
        self.linear3=LinearMap(indim=hiddenlayers[1], outdim=outdim, alpha=alpha, lr=lr)

    def forward(self, x, train=True):
        """
        :param x: (indim, batch_size) input matrix
        :param train: optional param indicating training
        """
        linear1=self.linear1.forward(x)
        activate1=self.activation1.forward(linear1, train)
        linear2=self.linear2.forward(activate1)
        activate2=self.activation2.forward(linear2,train)
        result=self.linear3.forward(activate2)
        return result
    def backward(self, grad_wrt_out):
        """
        :param grad_wrt_out:
            (outdim, batch_size) gradient matrix from next Transform
        :return grad_wrt_x:
            (indim, batch_size) gradient matrix for previous Transform
        """
        linear3=self.linear3.backward(grad_wrt_out)
        activate2=self.activation2.backward(linear3)
        linear2=self.linear2.backward(activate2)
        activate1=self.activation1.backward(linear2)
        linear1=self.linear1.backward(activate1)
        return linear1

    def step(self):
        self.linear1.step()
        self.linear2.step()
        self.linear3.step()

    def zerograd(self):
        self.linear1.zerograd()
        self.linear2.zerograd()
        self.linear3.zerograd()

    def loadparams(self, Ws, bs):
        """
        Load parameters for autograder (follow similar steps to SingleLayerMLP)
        e.g. [LinearMap1.W, LinearMap2.W, ...]
        """
        self.linear1.loadparams(Ws[0], bs[0])
        self.linear2.loadparams(Ws[1], bs[1])
        self.linear3.loadparams(Ws[2], bs[2])


    def getWs(self):
        w_linear1=self.linear1.getW()
        w_linear2=self.linear2.getW()
        w_linear3=self.linear3.getW()
        return [w_linear1, w_linear2, w_linear3]

    def getbs(self):
        b_linear1=self.linear1.getb()
        b_linear2=self.linear2.getb()
        b_linear3=self.linear3.getb()
        return [b_linear1, b_linear2, b_linear3]


if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You can also use a Jupyter notebook that imports this implementation.

    NOTE:
    You MUST use your class implementations to train the model and to get the results.

    DO NOT use PyTorch or Tensorflow get the results.

    The results generated using these libraries will be different as
    compared to your implementation.
    """
    with np.load('omniglot_12.npz') as data:
        trainX = data['trainX']
        testX = data['testX']
        trainY = data['trainY']
        testY = data['testY']

    # for i in range(5):
    #     plt.imshow(trainX[i].reshape(105,105))
    #     plt.axis("off")
    #     plt.show()

    def get_loss_accuracy(x, labels, model):
        logits=model.forward(x, train=False)
        softmax_layer=SoftmaxCrossEntropyLoss()
        loss=softmax_layer.forward(logits, labels)
        accuracy=softmax_layer.getAccu()
        return (loss, accuracy)



    batch_size=32
    num_batches=np.shape(trainX)[0]//batch_size
    epochs=200
    indim=np.shape(trainX)[1]
    #number of labels
    outdim=12
    model1a=SingleLayerMLP(indim, outdim, hiddenlayer=40,alpha=0, dropout_probability=0, lr=0.001)
    model1b=SingleLayerMLP(indim, outdim, hiddenlayer=40,alpha=0.4, dropout_probability=0, lr=0.001)
    model1c=SingleLayerMLP(indim, outdim, hiddenlayer=40, alpha=0.4, dropout_probability=0.2, lr=0.001)
    model1d=SingleLayerMLP(indim, outdim, hiddenlayer=100, alpha=0.4, dropout_probability=0.2, lr=0.001)

    model2a=TwoLayerMLP(indim, outdim, hiddenlayers=[40,40],alpha=0, dropout_probability=0, lr=0.001)
    model2b=TwoLayerMLP(indim, outdim, hiddenlayers=[40,40],alpha=0.4, dropout_probability=0, lr=0.001)
    model2c=TwoLayerMLP(indim, outdim, hiddenlayers=[40,40], alpha=0.4, dropout_probability=0.2, lr=0.001)
    model2d=TwoLayerMLP(indim, outdim, hiddenlayers=[100,100], alpha=0.4, dropout_probability=0.2, lr=0.001)
    def training_loop(model):
        train_losses=[]
        test_losses=[]
        train_accuracies=[]
        test_accuracies=[]
        for i in range(epochs):
            print("current epochs", i)
            train_size=np.shape(trainX)[0]
            randomized_indices=np.random.permutation(train_size)
            shuffled_x=trainX[randomized_indices]
            shuffled_y=trainY[randomized_indices]

            for j in range(num_batches):
                curr_x_batch=shuffled_x[j*batch_size:(j+1)*batch_size,:]
                curr_y_batch=shuffled_y[j*batch_size:(j+1)*batch_size]

                curr_y_labels=labels2onehot(curr_y_batch)
                #Reset gradients
                model.zerograd()

                #Forward pass
                pre_softmax=model.forward(curr_x_batch.T)

                #Calculate loss and gradient
                softmax_layer=SoftmaxCrossEntropyLoss()
                loss=softmax_layer.forward(pre_softmax, curr_y_labels.T)

                #Backward pass
                grad_wrt_out=softmax_layer.backward()
                model.backward(grad_wrt_out)

                #Apply gradients
                model.step()
            train_labels=labels2onehot(trainY)
            test_labels=labels2onehot(testY)
            (train_loss, train_accuracy)=get_loss_accuracy(trainX.T, train_labels.T, model)
            (test_loss, test_accuracy)=get_loss_accuracy(testX.T, test_labels.T,model)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, test_losses, train_accuracies, test_accuracies

    train_losses1a, test_losses1a, train_accuracies1a, test_accuracies1a=training_loop(model1a)
    train_losses1b, test_losses1b, train_accuracies1b, test_accuracies1b=training_loop(model1b)
    train_losses1c, test_losses1c, train_accuracies1c, test_accuracies1c=training_loop(model1c)
    train_losses1d, test_losses1d, train_accuracies1d, test_accuracies1d=training_loop(model1d)
    # Plotting
    plt.figure()
    plt.plot(np.arange(200), train_losses1a, label='Train Loss (a)')
    plt.plot(np.arange(200), train_losses1b, label='Train Loss (b)')
    plt.plot(np.arange(200), train_losses1c, label='Train Loss (c)')
    plt.plot(np.arange(200), train_losses1d, label='Train Loss (d)')

    plt.plot(np.arange(200), test_losses1a, label='Test Loss (a)')
    plt.plot(np.arange(200), test_losses1b, label='Test Loss (b)')
    plt.plot(np.arange(200), test_losses1c, label='Test Loss (c)')
    plt.plot(np.arange(200), test_losses1d, label='Test Loss (d)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Single Layer model test and train loss over epochs')
    plt.legend()
    plt.savefig('SingleLayer_Loss.png')

    plt.figure()
    plt.plot(np.arange(200), train_accuracies1a, label='Train Accuracy (a)')
    plt.plot(np.arange(200), train_accuracies1b, label='Train Accuracy (b)')
    plt.plot(np.arange(200), train_accuracies1c, label='Train Accuracy (c)')
    plt.plot(np.arange(200), train_accuracies1d, label='Train Accuracy (d)')

    plt.plot(np.arange(200), test_accuracies1a, label='Test Accuracy (a)')
    plt.plot(np.arange(200), test_accuracies1b, label='Test Accuracy (b)')
    plt.plot(np.arange(200), test_accuracies1c, label='Test Accuracy (c)')
    plt.plot(np.arange(200), test_accuracies1d, label='Test Accuracy (d)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.title('Single Layer model test and train accuracies over epochs')
    plt.legend()
    plt.savefig('SingleLayer_Accuracies.png')

    train_losses2a, test_losses2a, train_accuracies2a, test_accuracies2a=training_loop(model2a)
    train_losses2b, test_losses2b, train_accuracies2b, test_accuracies2b=training_loop(model2b)
    train_losses2c, test_losses2c, train_accuracies2c, test_accuracies2c=training_loop(model2c)
    train_losses2d, test_losses2d, train_accuracies2d, test_accuracies2d=training_loop(model2d)

    # Plotting
    plt.figure()
    plt.plot(np.arange(200), train_losses2a, label='Train Loss (a)')
    plt.plot(np.arange(200), train_losses2b, label='Train Loss (b)')
    plt.plot(np.arange(200), train_losses2c, label='Train Loss (c)')
    plt.plot(np.arange(200), train_losses2d, label='Train Loss (d)')

    plt.plot(np.arange(200), test_losses2a, label='Test Loss (a)')
    plt.plot(np.arange(200), test_losses2b, label='Test Loss (b)')
    plt.plot(np.arange(200), test_losses2c, label='Test Loss (c)')
    plt.plot(np.arange(200), test_losses2d, label='Test Loss (d)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Double layer test and train loss over epochs')
    plt.legend()
    plt.savefig('Double_Layer_Loss.png')

    plt.figure()
    plt.plot(np.arange(200), train_accuracies2a, label='Train Accuracy (a)')
    plt.plot(np.arange(200), train_accuracies2b, label='Train Accuracy (b)')
    plt.plot(np.arange(200), train_accuracies2c, label='Train Accuracy (c)')
    plt.plot(np.arange(200), train_accuracies2d, label='Train Accuracy (d)')

    plt.plot(np.arange(200), test_accuracies2a, label='Test Accuracy (a)')
    plt.plot(np.arange(200), test_accuracies2b, label='Test Accuracy (b)')
    plt.plot(np.arange(200), test_accuracies2c, label='Test Accuracy (c)')
    plt.plot(np.arange(200), test_accuracies2d, label='Test Accuracy (d)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.title('Double Layer model test and train accuracies over epochs')
    plt.legend()
    plt.savefig('Double_Layer_Accuracies.png')









