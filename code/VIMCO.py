"""
The MIT License (MIT)
Copyright (c) 2016 Evan Archer

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
sys.path.append('lib/') 
from MinibatchIterator import *
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify

from sklearn.cluster import KMeans



class BuildModel():
    def __init__(self, 
                opt_params, # dictionary of optimization parameters
                gen_params, # dictionary of generative model parameters
                GEN_MODEL,  # class that inherits from GenerativeModel
                rec_params, # dictionary of approximate posterior ("recognition model") parameters
                REC_MODEL, # class that inherits from RecognitionModel
                xDim=2, # dimensionality of latent state
                yDim=2, # dimensionality of observations
                ):
        
        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)
        
        #---------------------------------------------------------
        ## actual model parameters
        self.X, self.Y = T.matrices('X','Y')   # symbolic variables for the data
        self.hsamp = T.tensor3('hsamp')

        self.xDim   = xDim
        self.yDim   = yDim
        
        # instantiate our prior & recognition models
        self.mrec   = REC_MODEL(rec_params, self.Y, self.xDim, self.yDim, srng=self.srng, nrng=self.nrng)
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim, srng=self.srng, nrng = self.nrng)

        self.isTrainingRecognitionModel = True;
        self.isTrainingGenerativeModel = True;
        
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
        return params        
        
    def compute_objective_and_gradients(self):
        nSamp = self.hsamp.shape[2]  # How many samples will we take per data point

        def Lhat(h):
            # evaluate the recognition model density Q_\phi(h_i | y_i)
            q_hgy = self.mrec.evalLogDensity(h)

            # evaluate the generative model density P_\theta(y_i , h_i)
            p_yh =  self.mprior.evaluateLogDensity(h, self.Y)

            return [p_yh, q_hgy] # nMiniBatchSize x xDim

        # compute the conditional L(h_j | h_{-j}) thing
        out, _ = theano.map(Lhat, sequences=[self.hsamp]) # nSamp x nMinibatchSize

        p_yh = out[0]
        q_hgy = out[1]
        ff = (p_yh-q_hgy)
        fmax = T.max(ff, axis=0, keepdims = True)
        f_hy = T.exp(ff - fmax)
        Lhat = T.log(f_hy.mean(axis=0)) + fmax

        sum_across_samples = f_hy.sum(axis=0)
        
        I = (1-T.eye(ff.shape[0])).astype('int32')
        def loo_subset(i):
            xi = ff[i.nonzero()]    
            return xi
        ff_subs,_ = theano.map(loo_subset, sequences = I)  
        ff_smax = T.max(ff_subs, axis=1, keepdims=True)
        hold_out = T.log(T.exp(ff_subs - ff_smax).mean(axis=1,keepdims=True)) + ff_smax
        
        Lhat_cv = T.log(sum_across_samples/nSamp) + fmax - hold_out
        the_ws = f_hy / sum_across_samples

        weighted_q = T.sum((Lhat_cv*q_hgy + the_ws*(p_yh-q_hgy)).mean(axis=1))

        # gradients for approximate posterior
        dqhgy = T.grad(cost=weighted_q, wrt = self.mrec.getParams(), consider_constant=[the_ws,Lhat_cv])

        # gradients for prior
        dpyh = T.grad(cost=T.sum((the_ws*(p_yh-q_hgy)).mean(axis=1)), wrt = self.mprior.getParams(), consider_constant=[the_ws])
        
        return [Lhat.mean(), dpyh, dqhgy, ff, T.log(hold_out+eps)+fmax]


    def update_params(self, grads, L):
        batch_y = T.matrix('batch_y')
        h = T.tensor3('h')
        lr = T.scalar('lr')

        # SGD updates
        #updates = [(p, p + lr*g) for (p,g) in zip(self.getParams(), grads)]

        # Adam updates        
        # We negate gradients because we formulate in terms of maximization.
        updates = lasagne.updates.adam([-g for g in grads], self.getParams(), lr) 
        #, 1 - self.b1, 1 - self.b2, self.e)

        perform_updates_params = theano.function(
                 outputs=L,
                 inputs=[ theano.In(batch_y), theano.In(h), theano.In(lr)],
                 updates=updates,
                 givens={
                     self.Y: batch_y,
                     self.hsamp: h
                 }
             )

        return perform_updates_params

    def fit(self, y_train, batch_size = 50, max_epochs=100, learning_rate = 3e-4, nSamp = 5):
        
        train_set_iterator = DatasetMiniBatchIterator(y_train, batch_size)
        
        [Lhat, dpyh, dqhgy, _, _] = self.compute_objective_and_gradients()
        
        param_updater = self.update_params(dpyh+dqhgy,Lhat)

        # set dummy sampling variable to 0's
        hsamp_np = np.zeros([nSamp, batch_size, self.xDim]).astype(theano.config.floatX)

        avg_costs = []
        
        epoch = 0
        while epoch < max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for y in train_set_iterator:
                for idx in np.arange(nSamp):
                    hsamp_np[idx] = self.mrec.getSample(y)
                avg_cost = param_updater(y, hsamp_np, learning_rate)
                if np.mod(batch_counter, 10) == 0:
                    print '(L): (%f)\n' % (avg_cost)
                avg_costs.append(avg_cost)
                batch_counter += 1
            epoch += 1

        return avg_costs
