# Common imports
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.cluster import KMeans

# Stheno related imports
import lab.tensorflow as B
from stheno import GP, EQ
from varz.tensorflow import Vars, minimise_l_bfgs_b

# Import NSEQ kernel
from nseq import NSEQ

class NSGPRegression:
    def __init__(self, X, y, num_inducing_points, f, vs, seed=0):
        self.num_inducing_points = num_inducing_points
        self.X = X
        self.y = y
        self.vs = vs

        assert len(X.shape) == 2
        assert len(y.shape) == 2
        self.input_dim = X.shape[1]
        B.random.set_random_seed(seed)
        self.X_bar = f(self.X, num_inducing_points) # f to select inducing points
        
    def init_params(self, seed):
        B.random.set_random_seed(seed)
        
        self.vs.positive(init=, shape=, name='local_std')

    def LocalGP(self, vs, X, return_logpdf=False): # Getting lengthscales for entire train_X (self.X)
        l_list = []
        if return_logpdf:
            log_pdf_list = []
        for dim in range(self.input_dim):
            f = GP(vs.positive(init = B.rand()+1, name='local_std')**2 *\
                EQ().stretch(vs.positive(B.rand()+1, name='local_gp_ls'+str(dim))))
            f_post = f | (f(self.X_bar[:, dim]), 
                          vs.positive(B.rand(self.num_inducing_points,1)+1, 
                                    shape=(self.num_inducing_points,1), name='local_ls'+str(dim)))
            l = f_post(X[:, dim]).mean.mat
            l_list.append(l)
            if return_logpdf:
                log_pdf_list.append(f(self.X_bar[:, dim], ))
        
        return l_list
    
    def GlobalGP(self, vs): # Construct global GP and return nlml
        l_list = self.LocalGP(vs, self.X)
        global_ls = tf.concat(l_list, axis=1)
        
        f = GP(vs.positive(B.rand()+1, name='global_std')**2 * NSEQ(global_ls, global_ls))
        
        return -f(self.X, vs.positive(B.rand()+1, name='global_noise')**2).logpdf(self.y)
    
    def optimize(self, iters=1000, jit=False, trace=False): # Optimize hyperparams
        self.vs = Vars(tf.float64)
        minimise_l_bfgs_b(self.GlobalGP, self.vs, trace=trace, jit=jit, iters=iters)
        # self.vs.print()
        
    def predict(self, X_new): # Predict at new locations
        l_list = self.LocalGP(self.vs, self.X)
        global_ls = tf.concat(l_list, axis=1)
        
        l_list_new = self.LocalGP(self.vs, X_new)
        global_ls_new = tf.concat(l_list_new, axis=1)
        
        X_scaled = self.X/global_ls
        X_new_scaled = X_new/global_ls_new
        
        K = self.vs['global_std']**2 * B.exp(-B.sum((X_scaled[:,None,:] - X_scaled[None,:,:])**2, axis=2))
        K_star = self.vs['global_std']**2 * B.exp(-B.sum((X_new_scaled[:,None,:] - X_scaled[None,:,:])**2, axis=2))
        K_star_star = self.vs['global_std']**2 * B.exp(-B.sum((X_new_scaled[:,None,:] - X_new_scaled[None,:,:])**2, axis=2))
        
        L = B.cholesky(K + B.eye(self.X.shape[0]) * self.vs['global_noise']**2)
        alpha = B.cholesky_solve(L, self.y)
        
        pred_mean = K_star@alpha
        
        v = B.cholesky_solve(L, B.T(K_star))
        pred_var = K_star_star + B.eye(X_new.shape[0])*self.vs['global_noise']**2 - K_star@v
        
        return pred_mean, pred_var