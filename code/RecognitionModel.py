import sys
sys.path.append('../lib/')

import theano
import lasagne
from theano import tensor as T, function, printing
import theano.tensor.nlinalg as Tla
import numpy as np
import theano.tensor.slinalg as Tsla

class RecognitionModel(object):
    '''
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    '''

    def __init__(self,Input,indices,xDim,yDim,srng = None,nrng = None):
        self.srng = srng
        self.nrng = nrng

        self.xDim = xDim
        self.yDim = yDim
        self.Input = Input
        self.indices = indices

    def evalEntropy(self):
        '''
        Evaluates entropy of posterior approximation

        H(q(x))

        This is NOT normalized by the number of samples
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def getParams(self):
        '''
        Returns a list of Theano objects that are parameters of the
        recognition model. These will be updated during learning
        '''
        return self.params

    def getSample(self):
        '''
        Returns a Theano object that are samples from the recognition model
        given the input
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def setTrainingMode(self):
        '''
        changes the internal state so that `getSample` will possibly return
        noisy samples for better generalization
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def setTestMode(self):
        '''
        changes the internal state so that `getSample` will supress noise
        (e.g., dropout) for prediction
        '''
        raise Exception("Please implement me. This is an abstract method.")


class GMMRecognition(RecognitionModel):

    def __init__(self,RecognitionParams,Input,xDim,yDim,srng = None,nrng = None):
        '''
        h = Q_phi(x|y), where phi are parameters, x is our latent class, and y are data
        '''
        super(GMMRecognition, self).__init__(Input,None,xDim,yDim,srng,nrng)
        self.N = Input.shape[0]
        self.NN_h = RecognitionParams['NN_Params']['network']
        self.h = lasagne.layers.get_output(self.NN_h, inputs = self.Input)

    def getParams(self):
        network_params = lasagne.layers.get_all_params(self.NN_h)
        return network_params

    def getSample(self, Y):
        pi=T.clip(self.h, 0.001, 0.999).eval({self.Input:Y})
        pi= (1/pi.sum(axis=1))[:, np.newaxis]*pi #enforce normalization (undesirable; for numerical stability)
        x_vals = np.zeros([pi.shape[0],self.xDim])
        for ii in xrange(pi.shape[0]):
            x_vals[ii,:] = np.random.multinomial(1, pi[ii], size=1) #.nonzero()[1]

        return x_vals.astype(bool)

    def evalLogDensity(self, hsamp):

        ''' We assume each sample is a single multinomial sample from the latent h, so each sample is an integer class.'''
        return T.log((self.h*hsamp).sum(axis=1))
