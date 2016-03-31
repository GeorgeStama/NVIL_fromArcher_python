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

import numpy as np

class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, y, batch_size=100):
        self.y = y
        self.batch_size = batch_size
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

