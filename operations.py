import numpy as np

# Attension:
# - Never change the value of input, which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operation):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operation):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class fc(operation):
    def __init__(self):
        super(fc, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of fc layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


class conv(operation):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad = 2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        output = None

        #########################################

        
        batch, _, in_height, in_width = input.shape

        wid_out = 1+ (input.shape[3] + 2 * pad - kernel_w) // stride  # 2D conv outshape

        hei_out = 1+ (input.shape[2] + 2 * pad - kernel_h) // stride  # 2D conv outshape
        
        in_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        
        from utils import tools

        img = tools.img2col(in_pad, [i*stride for i in range(hei_out)],
                                 [i*stride for i in range(wid_out)], kernel_h, kernel_w)
        
        output = (np.matmul(weights.reshape(out_channel, -1), img.transpose(1, 2, 0)
                           .reshape(in_channel * kernel_h * kernel_w, -1)) + bias.reshape(-1, 1))\
            .reshape(out_channel, hei_out, wid_out, batch).transpose(3, 0, 1, 2)
        
   

    
        #########################################

        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
             weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  
        kernel_w = self.conv_params['kernel_w']  
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        in_grad = None
        w_grad = None
        b_grad = None

        #########################################

        
       
        
        
        in_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        
        wid_out = 1+ (input.shape[3] + 2 * pad - kernel_w) // stride  # 2D conv outshape

        hei_out = 1+ (input.shape[2] + 2 * pad - kernel_h) // stride  # 2D conv outshape

        batch, _, hei_out, wid_out = out_grad.shape
        
        p=pad
        
        from utils import tools
        
        b_grad = np.sum(out_grad, axis=(0, 2, 3))
        
        img = tools.img2col(in_pad, [i * stride for i in range(hei_out)],
                                 [i * stride for i in range(wid_out)], kernel_h, kernel_w)
        
        w_grad = (out_grad.transpose(1, 2, 3, 0).reshape(out_channel, -1) @ img.transpose(1, 2, 0)
                  .reshape(in_channel*kernel_h*kernel_w, -1).T).reshape(weights.shape)

        reshape_col = weights.reshape(out_channel, -1).T @ out_grad.transpose(1, 2, 3, 0).reshape(out_channel, -1)

        reshape_col2 =reshape_col .reshape(in_channel * kernel_w * kernel_h, -1, batch).transpose(2, 0, 1)

        in_grad = np.zeros(in_pad.shape, dtype=w_grad.dtype)
        
        c = np.array([[c for c in range(in_channel) for _ in range(kernel_h * kernel_w)] for _ in range(hei_out * wid_out)])

        i = np.array(
            [[i * stride + k // kernel_w for _ in range(in_channel) for k in range(kernel_h * kernel_w)] for i in range(hei_out)
             for _ in range(wid_out)])

        j = np.array(
            [[j * stride + k % kernel_w for _ in range(in_channel) for k in range(kernel_w * kernel_h)] for _ in range(hei_out) for
             j in range(wid_out)])

        np.add.at(in_grad, (slice(np.newaxis), c.T, i.T, j.T), reshape_col2)
        
  

      
        
        #########################################

        return in_grad, w_grad, b_grad


class pool(operation):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad = 2 means a 2-pixel border of padding with zeros.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        output = None

        #########################################
   
        
        
        p=pad 
        
        st=stride 

        in_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        
        batch, in_channel, in_height, in_width = input.shape

        wid_out = (input.shape[3] + 2 * p - pool_width) // st + 1

        hei_out = (input.shape[2] + 2 * p - pool_height) // st + 1
        
        in_grad = np.zeros_like(in_pad)
        
        pool_size = pool_height * pool_width
        
        
        for b in range(batch):
            
             for c in range(in_channel):
                
                 for h in range(0, hei_out):
                        
                    for w in range(0, wid_out):
             
                         if pool_type == 'max':
                    
                            output = np.max(input[b, c, h * st:h * st + pool_height, w * st:w * st + pool_width])
                        
                         elif pool_type == 'avg':
                                
                            output = out_grad[b, c, h, w]/pool_siz
        
        import itertools
        
        ip= itertools.product(range(hei_out), range(wid_out))

        output = np.array(list(map(lambda idx: in_pad[:, :, (idx[0] * stride):(idx[0] * stride) + pool_height,
                                               (idx[1] * stride):(idx[1] * stride) + pool_width],
                                  ip))).reshape(hei_out*wid_out, batch, in_channel, -1)
        
        if pool_type == 'max':
            output = np.max(output, axis=3).transpose(1, 2, 0).reshape(batch, in_channel, hei_out, -1)
        elif pool_type == 'avg':
            output = np.mean(output, axis=3).transpose(1, 2, 0).reshape(batch, in_channel, hei_out, -1)
       
       
            
        #########################################
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        in_grad = None

        #########################################
     
        p=pad 
        
        st=stride 
        
        in_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
   
        wid_out = (input.shape[3] + 2 * p - pool_width) // st + 1

        hei_out = (input.shape[2] + 2 * p - pool_height) // st + 1
        
        batch, in_channel, in_height, in_width = input.shape
        
        in_grad = np.zeros_like(in_pad)
        
        pool_size = pool_height * pool_width
        
             
        for b in range(batch):
            
             for c in range(in_channel):
                
                 for h in range(0, hei_out):
                        
                    for w in range(0, wid_out):
             
                         if pool_type == 'max':
                              
                            input_pool = input[b, c, h*st:h*st + pool_height, w * st:w * st + pool_width] 
                            
                            input_mask = input_pool == np.max(input_pool) 
                            
                            in_grad[b, c, h*st:h*stride + pool_height, w * st:w * st + pool_width] += out_grad[b, c, h, w] * input_mask
       
                         elif pool_type == 'avg':
             
                            in_grad[b, c, h*st + pool_height, w*st + pool_width] += out_grad[b, c, h, w]/pool_siz
      
       
            
        
        #########################################

        return in_grad


class dropout(operation):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        output = None
        if self.training:
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            #########################################
            # code here
            
            self.mask = p > self.rate
            output = input * self.mask / (1 - self.rate)
            
            #########################################
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            #########################################
        
            
            in_grad = out_grad * self.mask / (1 - self.rate)
           
            #########################################
        else:
            in_grad = out_grad
        return in_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
