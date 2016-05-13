import theano
import lasagne
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import theano.tensor.slinalg as Tsla
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

def NormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    return T.exp(logNormalPDFmat(X,Mu,XChol,xDim))

def logNormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    Lambda = Tla.matrix_inverse(T.dot(XChol,T.transpose(XChol)))
    XMu    = X-Mu
    return (-0.5 * T.dot(XMu, T.dot(Lambda,T.transpose(XMu)))
                  +0.5 * X.shape[0] * T.log(Tla.det(Lambda))
                  -0.5 * np.log(2*np.pi) * X.shape[0]*xDim)

def NormalPDF(X,Mu,XChol):
    return T.exp(logNormalPDF(X,Mu,XChol))

def logNormalPDF(X,Mu,XChol):
    Lambda = Tla.matrix_inverse(T.dot(XChol,T.transpose(XChol)))
    XMu    = X-Mu
    return (-0.5 * T.dot(XMu, T.dot(Lambda,T.transpose(XMu)))
                  +0.5 * T.log(Tla.det(Lambda))
                  -0.5 * np.log(2*np.pi) * X.shape[0])


class GenerativeModel(object):
    '''
    Interface class for generative time-series models
    '''
    def __init__(self,GenerativeParams,xDim,yDim,srng = None,nrng = None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.srng = srng
        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = T.matrix('Xsamp')

    def evaluateLogDensity(self):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        '''
        generates joint samples
        '''
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"


class MixtureOfGaussians(GenerativeModel):
    '''
    xDim - # classes
    yDim - dimensionality of observations
    '''
    def __init__(self, GenerativeParams, xDim, yDim, srng = None, nrng = None):

        super(MixtureOfGaussians, self).__init__(GenerativeParams,xDim,yDim,srng,nrng)

        # Mixture distribution
        if 'pi' in GenerativeParams:
            self.pi_un  = theano.shared(value=np.asarray(GenerativeParams['pi'], dtype = theano.config.floatX), name='pi_un', borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.pi_un  = theano.shared(value=np.asarray(100*np.ones(xDim), dtype = theano.config.floatX), name='pi_un' ,borrow=True)     # cholesky of observation noise cov matrix
        self.pi = self.pi_un/(self.pi_un).sum()

        if 'RChol' in GenerativeParams:
            self.RChol  = theano.shared(value=np.asarray(GenerativeParams['RChol'], dtype = theano.config.floatX), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol  = theano.shared(value=np.asarray(np.random.randn(xDim, yDim, yDim)/5, dtype = theano.config.floatX), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix

        if 'x0' in GenerativeParams:
            self.mu     = theano.shared(value=np.asarray(GenerativeParams['mu'], dtype = theano.config.floatX), name='mu',borrow=True)     # set to zero for stationary distribution
        else:
            self.mu     = theano.shared(value=np.asarray(np.random.randn(xDim, yDim), dtype = theano.config.floatX), name='mu', borrow=True)     # set to zero for stationary distribution


    def sampleXY(self,_N):

        _mu = np.asarray(self.mu.eval(), dtype = theano.config.floatX)
        _RChol = np.asarray(self.RChol.eval())
        _pi = T.clip(self.pi, 0.001, 0.999).eval()

        b_vals = np.random.multinomial(1, _pi, size=_N)
        x_vals = b_vals.nonzero()[1]

        y_vals = np.zeros([_N, self.yDim])
        for ii in xrange(_N):
            y_vals[ii] = np.dot(np.random.randn(1,self.yDim), _RChol[x_vals[ii],:,:].T) + _mu[x_vals[ii]]

        b_vals = np.asarray(b_vals,dtype = theano.config.floatX)
        y_vals = np.asarray(y_vals,dtype = theano.config.floatX)

        return [b_vals, y_vals]

    def getParams(self):
        return [self.RChol] + [self.mu] + [self.pi_un]

    def evaluateLogDensity(self,h,Y):
        X = h.nonzero()[1]
        LogDensityVec,_ = theano.map(logNormalPDF, sequences = [Y,self.mu[X],self.RChol[X]])
        return LogDensityVec + T.log(self.pi[X])
