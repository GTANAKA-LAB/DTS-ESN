
#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################
# esn_dts_closedloop.py: DTS-ESN (closed-loop model)
# (c) 2021 Gouhei Tanaka
# Citation:
# Tanaka et al., "Reservoir computing with diverse timescales for 
# prediction of multiscale dynamics", arXiv:2108.09446
#################################################################

import numpy as np
import networkx as nx


def identity(x):
    return x


class Input:
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: input dim
        param N_x: reservoir size
        param input_scale: input scaling
        '''
        # uniform distribution
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))
        
    # weighted sum
    def __call__(self, u):
        '''
        param u: (N_u)-dim vector
        return: (N_x)-dim vector
        '''
        return np.dot(self.Win, u)


class Reservoir:
    def __init__(self, N_x, density, rho, activation_func, leaking_rate,
                 seed=0):
        '''
        param N_x: reservoir size
        param density: connection density
        param rho: spectral radius
        param activation_func: activation function
        param leaking_rate: leak rates
        param seed
        '''
        self.seed = seed
        self.N_x = N_x
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)
        self.activation_func = activation_func
        self.alpha = leaking_rate
        self.saver = np.zeros(N_x)
        
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyi random graph
        m = int(N_x*(N_x-1)*density/2)
        G = nx.gnm_random_graph(N_x, m, self.seed)
        connection = nx.to_numpy_matrix(G)
        W = np.array(connection)

        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # rescaling
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        W *= rho / sp_radius

        return W

    def __call__(self, x_in):
        '''
        param x_in: x before update
        return: x after update
        '''
        #self.x = self.x.reshape(-1, 1)
        self.x = np.multiply(1.0 - self.alpha, self.x) + np.multiply(self.alpha, self.activation_func(np.dot(self.W, self.x) + x_in))

        return self.x

    def reset_reservoir_state(self):
        self.x *= 0.0

    def savestate(self):
        self.saver = self.x

    def loadstate(self):
        self.x = self.saver 


class Output:
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_xu1: = N_x + N_u + 1
        param N_y: output dim
        param seed
        '''
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x))

    def __call__(self, x):
        '''
        param x: (N_x)-dim vector
        return: (N_y)-dim vector
        '''
        return np.dot(self.Wout, x)

    def setweight(self, Wout_opt):
        self.Wout = Wout_opt        

class Feedback:
    def __init__(self, N_y, N_x, fb_scale, seed=0):
        '''
        param N_y: output dim
        param N_x: reservoir size
        param fb_scale: feedback scaling
        param seed
        '''
        np.random.seed(seed=seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))
        
    def __call__(self, y):
        '''
        param y: (N_y)-dim vector
        return: (N_x)-dim vector
        '''
        return np.dot(self.Wfb, y)


class Pseudoinv:
    def __init__(self, N_x, N_y):
        '''
        param N_x
        param N_y
        '''
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))
        
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))
        
    def get_Wout_opt(self):
        Wout_opt = np.dot(self.D, np.linalg.pinv(self.X))
        return Wout_opt


class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        '''
        param N_x: reservoir size
        param N_y: output dim
        param beta: regularization factor
        '''
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT \
                                     + self.beta*np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)

        return Wout_opt


class ESN:
    def __init__(self, N_u, N_y, N_x, density=0.05, input_scale=1.0,
                 rho=0.95, activation_func=np.tanh, fb_scale = None,
                 fb_seed=0, noise_level = None, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity,
                 classification = False, average_window = None, seed=0):
        '''
        param N_u: input dim
        param N_y: output dim
        param N_x: reservoir size
        param density: connection density
        param input_scale: input scaling factor
        param rho: spectral radius
        param activation_func: activation function
        param fb_scale: feedback scaling
        param fb_seed
        param leaking_rate: leak rates
        param output_func: activation function in the readout
        param inv_output_func: inverse of output_func
        param classification: True if classification problem (default: False)
        param average_window: window size (default: None)
        '''
        self.seed=seed
        self.Input = Input(N_u, N_x, input_scale, seed=self.seed)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate, seed=self.seed)
        self.Output = Output(N_x, N_y, seed=self.seed)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.y_prev_after_train = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification

        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=self.seed)
            self.noise = np.random.uniform(-noise_level, noise_level, (self.N_x, 1))

        if classification:
            if average_window is None:
                raise ValueError('Window for time average is not given!')
            else:
                self.window = np.zeros((average_window, N_x))

    def train(self, U, D, optimizer, trans_len = None):
        '''
        U: input data
        D: desired output
        optimizer: readout algorithm
        trans_len: length of transient period
        return: model output before training (Time x N_y)
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []

        for n in range(train_len):
            x_in = self.Input(U[n])

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            d = D[n]
            d = self.inv_output_func(d)

            if n > trans_len:
                optimizer(d, x)

            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d

        self.Output.setweight(optimizer.get_Wout_opt())

        self.y_prev_after_train = self.y_prev
        #print('train', self.y_prev_after_train)

        self.Reservoir.savestate()
        
        return np.array(Y)

    def predict(self, U):
        test_len = len(U)
        Y_pred = []

        for n in range(test_len):
            x_in = self.Input(U[n])

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            x = self.Reservoir(x_in)

            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        return np.array(Y_pred)

    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]
        self.y_prev = self.y_prev_after_train
        
        for n in range(test_len):
            x_in = self.Input(y)

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            x = self.Reservoir(x_in)

            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)
    
