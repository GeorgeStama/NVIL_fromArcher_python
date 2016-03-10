"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer & Josh Merel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    

from GenerativeModel import *
from RecognitionModel import *
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify

from sklearn.cluster import KMeans

class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, y, batch_size=100):
        self.y = y
        self.batch_size = batch_size
        from sklearn.utils import check_random_state
        self.rng = np.random.RandomState(20150503)

    def __iter__(self):
        n_samples = self.y.shape[0]
        if n_samples == self.batch_size:
            yield self.y
        for _ in xrange(n_samples / self.batch_size):
            if self.batch_size > 1:
                i = int(self.rng.rand(1) * (n_samples-self.batch_size-1))
            else:
                i = int(math.floor(self.rng.rand(1) * n_samples))
            ii = np.arange(i, i + self.batch_size)                    
            yield self.y[ii] 

class BuildModel():
    def __init__(self, 
                opt_params, # dictionary of optimization parameters
                gen_params, # dictionary of generative model parameters
                GEN_MODEL,  # class that inherits from GenerativeModel
                rec_params, # dictionary of approximate posterior ("recognition model") parameters
                REC_MODEL, # class that inherits from RecognitionModel
                xDim=2, # dimensionality of latent state
                yDim=2, # dimensionality of observations
                nCUnits = 100 # number of units used in the (single-layer) bias-correction network
                ):
        
        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)
        
        #---------------------------------------------------------
        ## actual model parameters
        self.X, self.Y = T.matrices('X','Y')   # symbolic variables for the data
        self.hsamp = T.lmatrix('hsamp')

        self.xDim   = xDim
        self.yDim   = yDim
        
        # instantiate our prior & recognition models
        self.mrec   = REC_MODEL(rec_params, self.Y, self.xDim, self.yDim, srng=self.srng, nrng=self.nrng)
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim, srng=self.srng, nrng = self.nrng)

        self.isTrainingRecognitionModel = True;
        self.isTrainingGenerativeModel = True;
        
        # NVIL Bias-correction network
        C_nn = lasagne.layers.InputLayer((None, yDim))
        C_nn = lasagne.layers.DenseLayer(C_nn, nCUnits, nonlinearity=leaky_rectify, W=lasagne.init.Orthogonal())
        self.C_nn = lasagne.layers.DenseLayer(C_nn, 1, nonlinearity=linear, W=lasagne.init.Orthogonal())
        
        self.c = theano.shared(value = np.asarray(opt_params['c0'], dtype=theano.config.floatX))
        self.v = theano.shared(value = np.asarray(opt_params['v0'], dtype=theano.config.floatX))
        self.alpha = theano.shared(value = np.asarray(opt_params['alpha'], dtype=theano.config.floatX))
        # ADAM defaults
        self.b1=0.1
        self.b2=0.001
        self.e=1e-8
        
    def getParams(self):
        ''' 
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []        
        params = params + self.mprior.getParams()            
        params = params + self.mrec.getParams()            
        params = params + lasagne.layers.get_all_params(self.C_nn)
        return params        
        
    def get_nvil_cost(self,Y,hsamp):
        '''
        NOTE: Y and hsamp are both assumed to be symbolic Theano variables. 
        
        '''
        
        # First, compute L and l (as defined in Algorithm 1 in Gregor & ..., 2014)
        
        # evaluate the recognition model density Q_\phi(h_i | y_i)
        q_hgy = self.mrec.evalLogDensity(hsamp)

        # evaluate the generative model density P_\theta(y_i , h_i)
        p_yh =  self.mprior.evaluateLogDensity(hsamp,Y)
        
        C_out = lasagne.layers.get_output(self.C_nn, inputs = Y).flatten()
        
        L = p_yh.mean() - q_hgy.mean()
        l = p_yh - q_hgy - C_out
        
        return [L,l,p_yh,q_hgy,C_out]
        
    def get_nvil_gradients(self,l,p_yh,q_hgy,C_out):
  
        def comp_param_grad(ii, pyh, qhgy, C, l, c, v):            
            lii = (l[ii] - c) / T.maximum(1,T.sqrt(v))            
            dpyh = T.grad(cost=pyh[ii], wrt = self.mprior.getParams())
            dqhgy = T.grad(cost=qhgy[ii], wrt =  self.mrec.getParams())
            dcx = T.grad(cost=C[ii], wrt = lasagne.layers.get_all_params(self.C_nn))
            output = [t for t in dpyh] + [t*lii for t in dqhgy] + [t*lii for t in dcx]
            return output
        
        grads,_ = theano.map(comp_param_grad, sequences = [T.arange(self.Y.shape[0])], non_sequences = [p_yh, q_hgy, C_out, l, self.c, self.v] )
        
        return [g.mean(axis=0, dtype=theano.config.floatX) for g in grads]
    
    def update_cv(self, l):
        batch_y = T.matrix('batch_y')
        h = T.lmatrix('h')
        
        # Now compute derived quantities for the update
        cb = l.mean(dtype = theano.config.floatX)
        vb = T.cast(l.var(), theano.config.floatX)
    
        updates = [(self.c, self.alpha * self.c + (1-self.alpha) * cb),
                   (self.v, self.alpha * self.v + (1-self.alpha) * vb)]
    
        perform_updates_cv = theano.function(
                 outputs=[self.c,self.v],
                 inputs=[ theano.Param(batch_y), theano.Param(h)],
                 updates=updates,
                 givens={
                     self.Y: batch_y,
                     self.hsamp: h
                 }
             )
        
        return perform_updates_cv

    def update_params(self, grads, L):
        batch_y = T.matrix('batch_y')
        h = T.lmatrix('h')
        lr = T.scalar('lr')
        
        # SGD updates
        #updates = [(p, p + lr*g) for (p,g) in zip(self.getParams(), grads)]
        
        # Adam updates        
        # We negate gradients because we formulate in terms of maximization.
        updates = lasagne.updates.adam([-g for g in grads], self.getParams(), lr) 
        #, 1 - self.b1, 1 - self.b2, self.e)
       
        perform_updates_params = theano.function(
                 outputs=L,
                 inputs=[ theano.Param(batch_y), theano.Param(h), theano.Param(lr)],
                 updates=updates,
                 givens={
                     self.Y: batch_y,
                     self.hsamp: h
                 }
             )
        
        return perform_updates_params
    
    def fit(self, y_train, batch_size = 50, max_epochs=100, learning_rate = 3e-4):
        
        train_set_iterator = DatasetMiniBatchIterator(y_train, batch_size)
    
        L,l,p_yh,q_hgy,C_out = self.get_nvil_cost(self.Y, self.hsamp)
        
        cv_updater = self.update_cv(l)
                
        grads = self.get_nvil_gradients(l,p_yh,q_hgy,C_out)
    
        param_updater = self.update_params(grads,L)

        avg_costs = []
        
        epoch = 0
        while epoch < max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for y in train_set_iterator:
                hsamp_np = self.mrec.getSample(y)
                cx,vx = cv_updater(y, hsamp_np) # update c,v
                avg_cost = param_updater(y, hsamp_np, learning_rate)
                if np.mod(batch_counter, 10) == 0:
                    print '(c,v,L): (%f,%f,%f)\n' % (np.asarray(cx), np.asarray(vx), avg_cost)
                avg_costs.append(avg_cost)
                batch_counter += 1
            epoch += 1
        return avg_costs
