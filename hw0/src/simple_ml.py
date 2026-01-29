import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 1. 读取X文件
    with gzip.open(image_filename, 'rb') as f:
        # a. 解析元数据 (验证码/图片数量/列数/行数)
        # '>IIII'代表读取4个big-endian的无符号整数
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"X meta-data:{magic}, {num}, {rows}, {cols}")
        # b. 读取剩余数据并转换为np数组
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
        # 错误示范：python的for循环无法直接修改循环的item，对于每个循环item执行计算的时候会创建一个副本
        # 所以此处对image进行计算操作会创建一个新的image变量，images中的原始数据并没有变化
            # for image in images:
            #     image = image / 255.0
        # 正确示范：使用索引像c语言一样遍历或者如下使用numpy特性
        images = (images / 255.0).astype(np.float32) # 注意numpy的隐式转换是转换为双精度浮点数
        X = images

    # 2. 读取y文件
    with gzip.open(label_filename, 'rb') as f:
        # a. 读取元数据
        magic, num = struct.unpack('>II', f.read(8))
        print(f"y meta-data:{magic}, {num}")
        label = np.frombuffer(f.read(), dtype=np.uint8)
        y = label
    
    # 3. 返回元组
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # 1. 计算归一化logits
    Z_exp = np.exp(Z)
    row_sums = Z_exp.sum(axis=1, keepdims = True) # keepdim=True代表保留维度不会坍缩
    Z_probs = Z_exp / row_sums # 此处触发广播机制，当两个数组最右侧维度一致/有一个为1时触发

    # 2. 根据y提取对应元素
    correct_probs = Z_probs[np.arange(Z_probs.shape[0]), y] 

    # 3. 计算loss
    loss = -np.log(correct_probs).mean()

    return loss                                                 
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 1. 计算迭代次数，注意传入的batch是batch_size而不是迭代次数
    iter_num = X.shape[0] // batch # 注意此处使用地板除法，否则迭代次数会是浮点数

    # 2. batch迭代
    for iter in range(iter_num):
        X_batch = X[iter * batch : (iter + 1) * batch]
        y_batch = y[iter * batch : (iter + 1) * batch]
        # 计算batch内梯度
        # 计算Z矩阵 (batch, num_classes)
        Z_origin = X_batch @ theta
        row_sums = np.exp(Z_origin).sum(axis=1, keepdims=True) # 归一化需要对exp后的logits
        Z = np.exp(Z_origin) / row_sums
        # 计算I_y矩阵 (batch, num_classes)
        I_y = np.zeros((batch, theta.shape[1])) # 创建数组的函数需要传入一个元组
        # 将特定元素置1的错误示范：下面的代码将索引部分的值取出来赋给了I_y
        # I_y = I_zeros[np.arange(y_batch), y_batch]
        I_y[np.arange(X_batch.shape[0]), y_batch] = 1 # 区别：直接修改特定的值
        batch_grad = X_batch.T @ (Z - I_y) / batch
        # theta更新
        theta -= lr * batch_grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 1. 计算迭代次数
    iter_num = X.shape[0] // batch

    # 2. batch迭代
    for iter in range(iter_num):
        X_batch = X[iter * batch : (iter + 1) * batch]
        y_batch = y[iter * batch : (iter + 1) * batch]
        # 计算中间值
        relu_output = X_batch @ W1
        relu_output[relu_output < 0] = 0 # 采用布尔掩码优化效率
        logits = relu_output @ W2
        exp_logits = np.exp(logits)
        row_sums = exp_logits.sum(axis = 1, keepdims=True)
        # 计算S
        S = exp_logits / row_sums
        # 计算I_y
        I_y = np.zeros((batch, W2.shape[1]))
        I_y[np.arange(X_batch.shape[0]), y_batch] = 1
        # 计算梯度
        # W2 Grad
        w2_grad = relu_output.T @ (S - I_y) / batch
        # W1 Grad
        relu_grad = relu_output.copy()
        relu_grad[relu_grad <= 0] = 0
        relu_grad[relu_grad > 0] = 1
        # 下面代码需要加上括号，强制先计算 (S - I_y) @ W2.T
        w1_grad = X_batch.T @ (relu_grad * ((S - I_y) @ W2.T)) / batch
        # 更新梯度
        W1 -= lr * w1_grad
        W2 -= lr * w2_grad
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
