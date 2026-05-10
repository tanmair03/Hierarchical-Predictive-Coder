# code for importing natural images dataset borrowed from Canva
# different sizes - each will contribute to a patch
# preprocessing includes whitening, log transform, and DOG
# patches
# Gaussian mask on patches

import numpy as np
import cv2
import glob
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self,scale=1, ksize=5,sigma1=1.3, sigma2=2.6, sigma_gauss=0.2,foldername = 'canva_env_pictures', n_images = 5, w_pix = 26, h_pix = 16, overlap = 5, stride = True):
        self.w_pix = w_pix
        self.h_pix = h_pix
        self.overlap = overlap
        self.scale = scale
        self.load_images(foldername, n_images) # load raw images
        self.preprocess(ksize, sigma1, sigma2)
        
        self.create_gauss_mask(sigma=sigma_gauss)
        if not stride:
            self.create_patches()
        else:
            self.create_patches_stride()

    def load_images(self, foldername, n_images):
        rawimages = []
        for i in range(n_images):
            image = cv2.imread("{}/image{}.jpg".format(foldername, i))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            rawimages.append(image)
        self.rawimages = rawimages

    def apply_DoG_filter(self, img,size, sigma_c, sigma_s, show_dog_only=False):
        #g1 = cv2.GaussianBlur(img, ksize, sigma1)
        #g2 = cv2.GaussianBlur(img, ksize, sigma2)
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2
        center   = np.exp(-r2 / (2 * sigma_c**2)) / (2 * np.pi * sigma_c**2)
        surround = np.exp(-r2 / (2 * sigma_s**2)) / (2 * np.pi * sigma_s**2)
        dog = center - surround
        dog_kernel = dog / np.abs(dog).sum()
        self.dog_filter = dog_kernel
        if show_dog_only:
            return dog_kernel
        else:
            return convolve(img, dog_kernel, mode='reflect')

    def gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))


    def preprocess(self,ksize, sigma1, sigma2):
        rawimages = self.rawimages
        filteredimages = []
        dogfilteredimages = []
        whitenedimages = []
        for image in rawimages:
            img = np.log(image + 1)
            img = (image - image.mean())#/(image.std() + 1e-8) # center-mean
            filteredimages.append(img)
            dogfilteredimages.append(self.apply_DoG_filter(img,ksize, sigma1, sigma2))
            #whitenedimages.append(self.whiten_image(dogfilteredimages[-1]))

        self.filteredimages = filteredimages
        self.dogfilteredimages = dogfilteredimages
        self.whitenedimages= whitenedimages

    def create_gauss_mask(self, sigma=0.2):
        """ Create gaussian mask. """
        width = self.h_pix
        mask = np.zeros(width * width)
        hw = width//2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                mask[j*width + i] = self.gauss(r, sigma=sigma)
        mask = np.array(mask)
        # Normalize
        mask = mask/np.max(mask)
        self.mask = mask
        return mask
    

    def create_patches_stride(self):
        filtered_images = self.dogfilteredimages
        patches_all = []

        for image_index, filtered_image in enumerate(filtered_images):
            h, w = filtered_image.shape
            print((w, h))

            size_w = w // self.w_pix
            y = 0
            while (h>y+self.h_pix):  # slide over y (stride = 1)
                for i in range(size_w):     # fixed stride over x
                    x = self.w_pix * i

                    patch = filtered_image[y:y+self.h_pix, x:x+self.w_pix]

                    # normalize per patch
                    patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
                    #patch = patch - cv2.GaussianBlur(patch, (5,5), 1.0)

                    # whitening 
                    patches_all.append(patch * self.scale)

                y +=1

        # convert once at the end
        self.patches = np.array(patches_all, dtype=np.float32)

    def whiten_image(self, img, epsilon=0.1, cutoff_low=2, cutoff_high=None):
        """
        Proper frequency domain whitening.
        
        Instead of assuming 1/f² falloff, measure the actual
        power spectrum and invert it directly.
        
        epsilon   : regularisation — prevents dividing by near-zero power
        cutoff_low: suppress frequencies below this (removes DC drift)
        cutoff_high: suppress frequencies above this (removes noise amplification)
        """
        img   = img.astype(np.float64)
        h, w  = img.shape

        # 2D FFT
        F       = np.fft.fft2(img)
        F_shift = np.fft.fftshift(F)

        # power at each frequency
        power   = np.abs(F_shift)**2

        # build radial frequency grid
        cy, cx  = h//2, w//2
        Y, X    = np.mgrid[-cy:h-cy, -cx:w-cx]
        R       = np.sqrt(X**2 + Y**2)

        # radially average the power spectrum
        R_int   = R.astype(int)
        max_r   = R_int.max()
        avg_pow = np.zeros(max_r + 1)
        counts  = np.zeros(max_r + 1)

        for r_val in range(max_r + 1):
            mask          = R_int == r_val
            avg_pow[r_val] = power[mask].mean() if mask.sum() > 0 else 1.0
            counts[r_val]  = mask.sum()

        # whitening filter: 1 / sqrt(power) at each frequency
        # sqrt because we want to flatten amplitude, not power
        whiten_1d = 1.0 / (np.sqrt(avg_pow) + epsilon * np.sqrt(avg_pow.max()))

        # low frequency cutoff — suppress DC and very low freq
        # these carry illumination changes, not structure
        whiten_1d[:cutoff_low] = 0.0

        # high frequency cutoff — suppress noise amplification
        if cutoff_high is not None:
            whiten_1d[cutoff_high:] = 0.0
        else:
            # default: suppress top 10% of frequencies
            whiten_1d[int(max_r * 0.9):] = 0.0

        # map back to 2D
        whiten_2d = whiten_1d[R_int]

        # apply whitening
        F_white   = F_shift * whiten_2d

        # back to spatial domain
        img_white = np.real(np.fft.ifft2(np.fft.ifftshift(F_white)))

        # renormalise
        img_white = (img_white - img_white.mean()) / (img_white.std() + 1e-8)

        return img_white


    def compute_zca_matrix(patches, epsilon=0.1):
        """
        Compute ZCA whitening matrix from a set of patches.
        
        patches : (n, d) array where d = patch height * width
        epsilon : regularisation, prevents amplifying near-zero variance directions
        
        returns : W_zca (d, d) whitening matrix
                mean  (d,)   patch mean
        """
        n, d   = patches.shape

        # centre
        mean   = patches.mean(axis=0)        # (d,)
        X      = patches - mean              # (n, d)

        # covariance
        cov    = (X.T @ X) / n              # (d, d)

        # eigendecomposition
        # eigh is faster and more stable than eig for symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(cov)

        # sort descending
        idx      = np.argsort(eigvals)[::-1]
        eigvals  = eigvals[idx]
        eigvecs  = eigvecs[:, idx]

        # ZCA matrix: V * diag(1/sqrt(eigval + eps)) * V^T
        # eps scaled to largest eigenvalue so it's relative, not absolute
        eps    = epsilon * eigvals[0]
        D_inv  = np.diag(1.0 / np.sqrt(eigvals + eps))
        W_zca  = eigvecs @ D_inv @ eigvecs.T    # (d, d)

        return W_zca, mean


    def apply_zca(patches, W_zca, mean):
        """
        Apply precomputed ZCA matrix to patches.
        patches : (n, d) or (d,) array
        """
        return (patches - mean) @ W_zca.T



    def create_patches(self):
        filtered_images = self.dogfilteredimages
        patches_all = np.array([])
        for image_index, filtered_image in enumerate(filtered_images):
            w = filtered_image.shape[1]
            h = filtered_image.shape[0]
            print((w,h))
            size_w = w // self.w_pix
            size_h = h // self.h_pix
            patches = np.empty((size_h * size_w, self.h_pix, self.w_pix), dtype=np.float32)
            for j in range(size_h):
                y = self.h_pix * j
                for i in range(size_w):
                    x = self.w_pix * i
                    patch = filtered_image[y:y+self.h_pix, x:x+self.w_pix]

                    patch = (patch - np.mean(patch))/(np.std(patch) + 1e-6)
                    #patch = patch - np.mean(patch)
                    #patch = patch / (np.std(patch) + 1e-6)
                    # (16, 26)
                    # print(patch.shape)
                    
                    index = j*size_w + i
                    patches[index,:,:] = patch * self.scale

                
            if image_index == 0:
                patches_all = patches
            else:
                patches_all = np.concatenate((patches_all, patches), axis = 0)

        self.patches = patches_all
        
    def get_images(self, patch_index):
        patch = self.patches[patch_index]
        return self.get_images_from_patch(patch)
    
    def get_images_sparse(self, patch_index):
        patch = self.patches[patch_index]
        return self.get_images_from_patch(patch, use_mask=False)
    
    def get_images_from_patch(self, patch, use_mask=True):
        images = []
        for i in range(3):
            x = self.overlap * i
            # Apply gaussian mask
            image = patch[:, x:x+self.h_pix].reshape([-1])
            if use_mask:
                image = image * self.mask
            images.append(image)
        return images
    

    def visualize_dog_kernel(self,savefile="figures/dog_kernel.jpg"):
        """
        Plot the DoG kernel itself and its 1D cross-section through the centre row.
        Useful for checking your sigma values make sense.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

        ax1.imshow(self.dog_filter, interpolation='nearest')
        ax1.set_title("DoG kernel (2D)")
        ax1.axis('off')

        mid = self.dog_filter.shape[0] // 2
        ax2.plot(self.dog_filter[mid], 'k-o', markersize=4)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax2.set_title("Centre row cross-section")
        ax2.set_xlabel("Pixel offset")
        ax2.set_ylabel("Kernel value")

        plt.tight_layout()
        plt.savefig(savefile, dpi=150)
        plt.show()
