# model 
import numpy as np
import os


class Model: 
    def __init__(self, iteration=500, k1= 0.005, k2_init=0.01, sigma_sq=1, sigma_sq_td=10, alpha1=1, alpha2=0.05, lambd1=0.02, lambd2=0.00001):
        self.iteration = iteration
        
        self.k1      = k1 # Learning rate for r
        self.k2_init = k2_init # Initial learning rate for U
        self.energy = []
        self.energy_l2 = []
        self.prior_U = []
        self.prior_r = []
        self.prior_uh = []
        self.prior_rh = []
        self.U_history = []
        self.sigma_sq    = sigma_sq  # Variance of observation distribution of I
        self.sigma_sq_td = sigma_sq_td # Variance of observation distribution of r
        self.alpha1      = alpha1  # Precision param of r prior    (var=1.0,  std=1.0)
        self.alpha2      = alpha2 # Precision param of r_td prior (var=20.0, std=4.5)
        self.lambd1      = lambd1 # Precision param of U prior    (var=50.0, std=7.1)
        self.lambd2      = lambd2 # Precision param of Uh prior
        
        
        U_scale = lambd1#0.1
        self.Us = lambd1*np.random.normal(size=(3,256,32), scale=1/self.lambd1) #(np.random.rand(3,256,32) - 0.5) * U_scale
        self.Uh = lambd2*np.random.normal(size=(96,128), scale=1/self.lambd2) #(np.random.rand(96,128)   - 0.5) * U_scale
        
        for j in range(3):
                norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                self.Us[j] /= norms
        norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
        self.Uh /= norms
        
        self.k2 = self.k2_init

        # Scaling parameter for learning rate of level2
        self.level2_lr_scale = 10.0



    def loss(self, images, rs, rh, sparse = False):
        # calculate the loss for the updated model i.e., how does the model perform with these updated weights
        # 
        E_prior_r = 0
        E_recon=0
        E_prior_U = 0
        E_prior_rh = 0
        E_prior_Uh = 0
        E_td = 0

        c=0
        for img in images:
            E_recon += np.sum((img - self.Us[c].dot(rs[32*c:32*(c+1)]))**2) / self.sigma_sq # recon error
            E_prior_U += np.sum(self.lambd1*(self.Us[c]**2))
            c+=1

        if sparse:
            E_prior_r += self.alpha1*np.sum(np.log(rs**2 + 1))
            E_prior_rh += self.alpha2*np.sum(np.log(rh**2 + 1))
        else:
            E_prior_r += np.sum(self.alpha1*(rs**2))
            E_prior_rh += np.sum(self.alpha2*(rh**2))

        E_prior_Uh += np.sum(self.lambd2*(self.Uh**2))
        E_td += np.sum((rs - self.Uh.dot(rh))**2)/ self.sigma_sq_td

        self.prior_r.append(E_prior_r)
        self.prior_rh.append(E_prior_rh)
        self.prior_U.append(E_prior_U)
        self.prior_uh.append(E_prior_Uh)
        self.energy.append(E_recon)
        self.energy_l2.append(E_td)
            # per image, divide into patches, infer r, and calc error, prior err over r,rh,u,uh, top-down err



    def apply_images(self, images, training):
        rs = np.zeros([96],  dtype=np.float64) # np.random.normal(scale=1/self.alpha1, size=(96)) #
        rh = np.zeros([128], dtype=np.float64) # np.random.normal(scale=1/self.alpha2,size=(128))#
        error_tds = np.zeros([96], dtype=np.float64)
        dr_history = []
        drh_history = []

        for i in range(self.iteration):
            # Loop for iterations

            # Calculate r_td
            r_tds = self.Uh.dot(rh) 
            dr_all = np.zeros([96])
            for j in range(len(images)):
                I = images[j]
                r    = rs[   32*j:32*(j+1)]
                r_td = r_tds[32*j:32*(j+1)]
                U  = self.Us[j]
                Ur = U.dot(r)
                
                error    = I - Ur
                error_td = r_td - r
                    
                dr = (self.k1/self.sigma_sq) * U.T.dot(error) + (self.k1/self.sigma_sq_td) * error_td - self.k1 * 2 * self.alpha1 * r
                
                
                    
                rs[32*j:32*(j+1)] += dr
                dr_all[32*j:32*(j+1)] = dr

                    
                error_tds[32*j:32*(j+1)] = error_td

            # Level2 update
            drh = (self.k1*self.level2_lr_scale / self.sigma_sq_td) * self.Uh.T.dot(-error_tds) \
                  - self.k1*self.level2_lr_scale * 2* self.alpha2 * rh
            rh += drh
            drh_history.append(drh)
            dr_history.append(dr_all)    

        if training: # loss calculation 
            self.loss(images, rs, rh)

            '''
            if training:
                
                dUh = ((self.k2 * self.level2_lr_scale / self.sigma_sq_td)
                        * np.outer(-error_tds, rh)
                    - self.k2 * self.level2_lr_scale * self.lambd2 * self.Uh)
                self.Uh += dUh
        
                for j in range(3):
                    dU = ((self.k2 / self.sigma_sq) * np.outer(error, r)
                        - self.k2 * self.lambd1   * U)
                    self.Us[j] += dU

                for j in range(3):
                    norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                    self.Us[j] /= norms

                norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
                self.Uh /= norms
            E_prior_U = 0
            for j in range(3):
                E += np.sum((images[j] - self.Us[j].dot(rs[32*j:32*(j+1)]))**2) / self.sigma_sq # recon error
                E_prior_U += np.sum(self.lambd1*(self.Us[j]**2))

            E_prior_r = np.sum(self.alpha1*(rs**2))
            E_prior_rh = np.sum(self.alpha2*(r**2))
            E_prior_Uh = np.sum(self.lambd2*(self.Uh**2))
            E2 = np.sum((rs - self.Uh.dot(rh)))/ self.sigma_sq_td


        self.prior_r.append(E_prior_r)
        self.prior_rh.append(E_prior_rh)
        self.prior_U.append(E_prior_U)
        self.prior_uh.append(E_prior_Uh)
        self.energy.append(E)
        self.energy_l2.append(E2)
        self.U_history.append(self.Us)
        
            
        bu_term = (self.k1/self.sigma_sq) * U.T.dot(error)
        td_term = (self.k1/self.sigma_sq_td) * error_td
        pr_term = self.k1 * 2*self.alpha1 * r

        print(f"bu={np.mean(np.abs(bu_term)):.5f} | "
            f"td={np.mean(np.abs(td_term)):.5f} | "
            f"prior={np.mean(np.abs(pr_term)):.5f} | "
            f"err_td={error_td.mean():.5f}")
        '''
        if not training:
            return rs, r_tds, rh, error_tds, dr_history, drh_history
        return rs, r_tds, rh, error_tds

    
    def apply_images_sparse(self, images, training, prior='cauchy'):

        rs        = np.zeros([96],  dtype=np.float64)
        rh        = np.zeros([128], dtype=np.float64)
        error_tds = np.zeros([96],  dtype=np.float64)
        dr_history = []
        drh_history = []
        # select prior derivative once outside the loop
        if prior == 'cauchy':
            def g_prime(r):
                return 2 * r / (1 + r**2)          # Rao & Ballard 1999
        elif prior == 'laplace':
            def g_prime(r):
                return np.sign(r)  
                             # L1 / Laplace
        elif prior == 'gaussian':
            def g_prime(r):
                return 2 * self.alpha1 * r          # original Gaussian (baseline)
        else:
            raise ValueError(f"Unknown prior: {prior}")

        for i in range(self.iteration):

            r_tds = self.Uh.dot(rh)   # (96,)
            #rs = np.clip(rs, -5, 5)
            #rh = np.clip(rh, -5, 5)
            #rs = rs / (np.linalg.norm(rs) + 1e-6)
            #rh = rh / (np.linalg.norm(rh) + 1e-6)
            old_rs = rs.copy()
            for j in range(3):
                I    = images[j]
                r    = rs[32*j:32*(j+1)]
                r_td = r_tds[32*j:32*(j+1)]

                U  = self.Us[j]
                Ur = (U.dot(r))
                
                error    = I - Ur
                error_td = r_td - r
                dr = ((self.k1 / self.sigma_sq)    *  (U.T.dot(error))
                    + (self.k1 / self.sigma_sq_td) *  error_td
                    -   self.k1 * self.alpha1      *  g_prime(r))   # ← swapped
                rs[32*j:32*(j+1)] += dr
                dr_history.append(dr)


                error_tds[32*j:32*(j+1)] = error_td
                
                
                
                
    
            drh = ((self.k1 * self.level2_lr_scale / self.sigma_sq_td)
                    * (self.Uh.T.dot(-error_tds))
                - self.k1 * self.level2_lr_scale * self.alpha2 * rh)
            rh += drh
            drh_history.append(drh)
        '''
        bu_term = (self.k1 / self.sigma_sq) * U.T.dot(error)
        td_term = (self.k1 / self.sigma_sq_td) * error_td
        pr_term = self.k1 * self.alpha1 * g_prime(r)

        print(f"bu={np.mean(np.abs(bu_term)):.5f} | "
            f"td={np.mean(np.abs(td_term)):.5f} | "
            f"prior={np.mean(np.abs(pr_term)):.5f}")
        '''
        if training: # loss calculation 
            self.loss(images, rs, rh, sparse=True)
            
        '''
        if training:
       
            
            dUh = ((self.k2 * self.level2_lr_scale / self.sigma_sq_td)
                    * np.outer(-error_tds, rh)
                - self.k2 * self.level2_lr_scale * self.lambd2 * self.Uh)
            self.Uh += dUh
        
            for j in range(3):
                dU = ((self.k2 / self.sigma_sq) * np.outer(error, r)
                    - self.k2 * self.lambd1   * U)
                self.Us[j] += dU

            for j in range(3):
                norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                self.Us[j] /= norms

            norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
            self.Uh /= norms
        
        E_prior_U = 0
        for j in range(3):
            E += np.sum((images[j] - self.Us[j].dot(rs[32*j:32*(j+1)]))**2) / self.sigma_sq # recon error
            E_prior_U += np.sum(self.lambd1*(self.Us[j]**2))

        E_prior_r = np.sum(self.alpha1*(rs**2))
        E_prior_rh = np.sum(self.alpha2*(r**2))
        E_prior_Uh = np.sum(self.lambd2*(self.Uh**2))
        E2 = np.sum((rs - self.Uh.dot(rh)))/ self.sigma_sq_td

        self.prior_r.append(E_prior_r)
        self.prior_rh.append(E_prior_rh)
        self.prior_U.append(E_prior_U)
        self.prior_uh.append(E_prior_Uh)
        self.energy.append(E)
        self.U_history.append(self.Us)
        self.energy_l2.append(E2)
        '''
    
        if not training:
            return rs, r_tds, rh, error_tds, dr_history, drh_history
        
        return rs, r_tds, rh, error_tds
        
    
    def update_weights(self,rs, r_tds, rh, error_tds, images,sparse=False):

        dUh = ((self.k2 / self.sigma_sq_td)
                * np.outer(-error_tds, rh)
            - self.k2 * self.lambd2 * self.Uh)
        #self.Uh += dUh
        dU = [None]*3
        for j in range(3):
            I    = images[j]
            r    = rs[32*j:32*(j+1)]
            r_td = r_tds[32*j:32*(j+1)]

            U  = self.Us[j]
            Ur = (U.dot(r))
            
            error    = I - Ur
            dU[j] = ((self.k2 / self.sigma_sq) * np.outer(error, r)
                - self.k2 * self.lambd1   * U)
            
            #self.Us[j] += dU

            #norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
            #self.Us[j] /= norms

        #norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
        #self.Uh /= norms


        return dU, dUh
    
    
    def train_sparse(self, dataset):
        self.k2 = self.k2_init
        
        patch_size = len(dataset.patches) # 2375

        for i in range(patch_size):
            # Loop for all patches
            images = dataset.get_images_sparse(i)
            rs, r_tds, rh, error_tds = self.apply_images_sparse(images, training=True)
            
            dU, dUh = self.update_weights(rs, r_tds, rh, error_tds, images, sparse=True)
            
            # weight update + normalise
            for j in range(3):
                self.Us[j] += dU[j]
                norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                self.Us[j] /= norms
            
            self.Uh += dUh
            norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
            self.Uh /= norms
            
            if i% 500 == 0:
                print("Sample"+str(i))
    
            if i % 200 == 0:
                # Decay learning rate for U
                self.k2 = self.k2 / 1.015

        print("train finished")
    

    def train_batch(self, dataset, batch_size=100, shuffle=True, sparse = False):

        # This is a function for running 1 epoch        
        # energy calc -> avg energy per mini-batch
        # mini-batch - split dataset into batches, and update weights only after inference over 1 batch
        # shuffle patches before training 
        # learning in a different function
        # 1 function call for 1 image

        self.k2 = self.k2_init
        
        patch_size = len(dataset.patches) # 550k

        if shuffle:
            indices = np.random.permutation(len(dataset.patches))
        else:
            indices = np.arange(0, len(dataset.patches),1)
        
        for n_batch in range(int(patch_size/batch_size)):
            dU = [None]*batch_size
            dUh = [None]*batch_size
            self.k2 = self.k2_init
            for i in range(batch_size):
                # Loop for all patches
                if sparse:
                    images = dataset.get_images_sparse(indices[n_batch*batch_size+i])
                    rs, r_tds, rh, error_tds = self.apply_images_sparse(images, training=True) # inference
                else:
                    images = dataset.get_images(indices[n_batch*batch_size+i])
                    rs, r_tds, rh, error_tds = self.apply_images(images, training=True) # inference

                if i % 40 == 0:
                    # Decay learning rate for U
                    #print("Batch "+str(n_batch))
                    self.k2 = self.k2 / 1.015


             # learning per mini-batch
            dU, dUh = self.update_weights(rs, r_tds, rh, error_tds, images, sparse=sparse)
            
            # weight update + normalise
            for j in range(3):
                self.Us[j] += dU[j]
                norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                #self.Us[j] /= norms
            self.Uh += dUh
            #norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
            #self.Uh /= norms
            #self.loss(dataset, indices, sparse=sparse)

            if n_batch % 50 == 0:
                    # Decay learning rate for U
                    print("Batch "+str(n_batch))
                    #self.k2 = self.k2 / 1.015

        print("train finished")



    def train(self, dataset):
        self.k2 = self.k2_init
        
        patch_size = len(dataset.patches) # 2375

        for i in range(patch_size):
            # Loop for all patches
            images = dataset.get_images(i)
            rs, r_tds, rh, error_tds = self.apply_images(images, training=True)
            
            dU, dUh = self.update_weights(rs, r_tds, rh, error_tds, images, sparse=False)
            
            # weight update + normalise
            for j in range(3):
                self.Us[j] += dU[j]
                norms = np.linalg.norm(self.Us[j], axis=0, keepdims=True) + 1e-6
                self.Us[j] /= norms
            self.Uh += dUh
            norms = np.linalg.norm(self.Uh, axis=0, keepdims=True) + 1e-6
            self.Uh /= norms
            if i% 500 == 0:
                print("Sample"+str(i))
    
            if i % 200 == 0:
                # Decay learning rate for U
                
                self.k2 = self.k2 / 1.015

        print("train finished")

    def reconstruct(self, r, level=1):
        if level==1:
            rs = r
        else:
            rh = r
            rs = self.Uh.dot(rh) # (96,)
            
        patch = np.zeros((16,26), dtype=np.float64)
        
        for i in range(3):
            r = rs[32*i:32*(i+1)]
            U = self.Us[i]
            Ur = U.dot(r).reshape(16,16)
            patch[:, 5*i:5*i+16] += Ur
        return patch

    def get_level2_rf(self, index):
        Uh0 = self.Uh[:,index][0:32]
        Uh1 = self.Uh[:,index][32:64]
        Uh2 = self.Uh[:,index][64:96]

        UU0 = self.Us[0].dot(Uh0).reshape((16,16))
        UU1 = self.Us[1].dot(Uh1).reshape((16,16))
        UU2 = self.Us[2].dot(Uh2).reshape((16,16))
        
        rf = np.zeros((16,26), dtype=np.float64)
        rf[:, 5*0:5*0+16] += UU0
        rf[:, 5*1:5*1+16] += UU1
        rf[:, 5*2:5*2+16] += UU2    
        return rf

    def save(self, dir_name):
        #if not os.path.exists(dir_name):
        #    os.makedirs(dir_name)
        file_path = dir_name

        np.savez_compressed(file_path,
                            self)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        file_path = os.path.join(dir_name+".npz")
        return np.load(file_path, allow_pickle=True)['arr_0'].item()
