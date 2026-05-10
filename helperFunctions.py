import numpy as np
import matplotlib.pyplot as plt

def plot_r_distribution(self, dataset, n_samples=100, sparse=False, savefile="figures/r_dist_dense.jpg"):
    """
    Plot the distribution of r values across many patches.
    Sparse coding should give a heavy-tailed (leptokurtic) distribution
    — most values near zero, few large values.
    Gaussian prior gives a roughly normal distribution.
    """
    all_rs = []

    for i in range(n_samples):
        
        if sparse:
            images = dataset.get_images_sparse(i)
            rs, _, _, _,_,_ = self.apply_images_sparse(images, training=False)
        else:
            images = dataset.get_images(i)
            rs, _, _, _,_,_  = self.apply_images(images, training=False)
        all_rs.append(rs)

    all_rs = np.array(all_rs).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # histogram of r values
    axes[0].hist(all_rs, bins=100, density=True, color='steelblue', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title("Distribution of r values")
    axes[0].set_xlabel("r value")
    axes[0].set_ylabel("density")
    #axes[0].set_xlim([-1, 1])
    # sparse: sharp peak at 0 with heavy tails
    # dense:  broad bell curve

    # log scale to see tails
    axes[1].hist(np.abs(all_rs), bins=100, density=True,
                 color='steelblue', alpha=0.7, log=True)
    axes[1].set_title("Distribution of |r| (log scale)")
    axes[1].set_xlabel("|r| value")
    axes[1].set_ylabel("log density")
    #axes[1].set_xlim([0, 1])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    # sparse: straight line on log scale = exponential tails
    # dense:  curved = Gaussian tails

    # print summary stats
    print(f"mean |r|  = {np.mean(np.abs(all_rs)):.4f}")
    print(f"std r     = {np.std(all_rs):.4f}")
    print(f"kurtosis  = {float(np.mean((all_rs - np.mean(all_rs))**4) / np.std(all_rs)**4):.2f}")
    # kurtosis > 3 means super-Gaussian (sparse)
    # kurtosis ≈ 3 means Gaussian (not sparse)
    # kurtosis < 3 means sub-Gaussian (too uniform)
    print(f"sparsity (|r|<0.1) = {np.mean(np.abs(all_rs) < 0.1):.2f}")
    print(f"sparsity (|r|<0.5) = {np.mean(np.abs(all_rs) < 0.5):.2f}")
    print(f"percentiles: "
          f"50%={np.percentile(np.abs(all_rs), 50):.3f} "
          f"75%={np.percentile(np.abs(all_rs), 75):.3f} "
          f"90%={np.percentile(np.abs(all_rs), 90):.3f} "
          f"99%={np.percentile(np.abs(all_rs), 99):.3f}")

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.show()

def plot_energy(energy_history, window=100, savefile="figures/dense_energy.jpg"):
    energy = np.array(energy_history)
    
    # rolling average
    smoothed = np.convolve(energy, np.ones(window)/window, mode='valid')
    steps    = np.arange(len(smoothed))

    fig, axes = plt.subplots(1, 1, figsize=(10,5))

    # linear scale — shows overall trend
    axes.plot(steps, smoothed, linewidth=1.5)
    axes.set_xlabel("Training patch")
    axes.set_ylabel("Free energy (smoothed)")
    axes.set_title(f"Training energy (rolling avg, window={window})")

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.show()

    print(f"Initial energy (avg first 100):  {np.mean(energy[:100]):.2f}")
    print(f"Final energy   (avg last  100):  {np.mean(energy[-100:]):.2f}")
    print(f"Reduction: {(1 - np.mean(energy[-100:])/np.mean(energy[:100]))*100:.1f}%")




def r_convergence(model, dataset, n=5, sparse = False, savefile = "figures/dense_r_convergence.jpg"):
    """
    Show input patch vs reconstruction side by side.
    If filters are learning, these should look similar.
    """
    fig, axes = plt.subplots(1,1, figsize=(8,5))
    #fig.suptitle("Left: input | Centre: reconstruction | Right: error")
    #axes = axes.flatten()
    norm_dr=[]
    for idx in range(n):
        
        if sparse:
            images = dataset.get_images_sparse(idx)
            rs, r_tds, rh, error_tds,dr,drh = model.apply_images_sparse(
                images, training=False)
        else:
            images = dataset.get_images(idx)
            rs, r_tds, rh, error_tds, dr, drh = model.apply_images(
                images, training=False)
        #print(dr[0].shape)
        dr = np.array(dr)
        norm_dr.append(np.linalg.norm(dr, axis=-1))
    norm_dr = np.array(norm_dr)
    mean_norm = np.mean(norm_dr, axis=0)
    std_norm = np.std(norm_dr, axis=0)
        
    axes.plot(mean_norm, c='k')#,  cmap='gray', interpolation='nearest')
    plt.fill_between(np.arange(mean_norm.size),mean_norm-std_norm,mean_norm+std_norm, alpha=0.2, facecolor='k')
    axes.set_title("Convergence of r over Inference iterations, n_patches="+str(n))
        #axes[idx].axis('off')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlabel("# iteration")
    axes.set_ylabel("Mean ||dr||")
    plt.savefig(savefile)

def plot_reconstructions(model, dataset, n=5, sparse = False, savefile = "figures/dense_recon.jpg"):
    """
    Show input patch vs reconstruction side by side.
    If filters are learning, these should look similar.
    """
    fig, axes = plt.subplots(2,n, figsize=(3*n, 8))
    #fig.suptitle("Left: input | Centre: reconstruction | Right: error")

    np.random.seed(0)
    idx_choose = np.random.randint(dataset.patches.shape[0], size=n)
    for idx in range(n):
        
        if sparse:
            images = dataset.get_images_sparse(idx_choose[idx])
            rs, r_tds, rh, error_tds,_,_ = model.apply_images_sparse(
                images, training=False)
        else:
            images = dataset.get_images(idx_choose[idx])
            rs, r_tds, rh, error_tds,_,_ = model.apply_images(
                images, training=False)

        # reconstruct centre patch (module 1)
        r     = rs[32:64]
        U     = model.Us[1]
        recon = U.dot(r).reshape(16, 16)
        orig  = images[1].reshape(16, 16)
        err   = orig - recon

        axes[0,idx].imshow(orig,  cmap='gray', interpolation='nearest')
        axes[0,idx].set_title(f"Input {idx_choose[idx]}")
        axes[0,idx].axis('off')

        axes[1,idx].imshow(recon, cmap='gray', interpolation='nearest')
        axes[1,idx].set_title(f"Recon (err={np.mean(err**2):.3f})")
        axes[1,idx].axis('off')

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.show()


def plot_rh_distribution(self, dataset, n_samples=100, sparse=False, savefile="figures/r_dist_dense.jpg"):
    """
    Plot the distribution of r values across many patches.
    Sparse coding should give a heavy-tailed (leptokurtic) distribution
    — most values near zero, few large values.
    Gaussian prior gives a roughly normal distribution.
    """
    all_rs = []

    for i in range(n_samples):
        
        if sparse:
            images = dataset.get_images_sparse(i)
            _, _,rs, _,_,_ = self.apply_images_sparse(images, training=False)
        else:
            images = dataset.get_images(i)
            _, _,rs, _ ,_,_= self.apply_images(images, training=False)
        all_rs.append(rs)

    all_rs = np.array(all_rs).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # histogram of r values
    axes[0].hist(all_rs, bins=100, density=True, color='steelblue', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title("Distribution of rh values")
    axes[0].set_xlabel("rh value")
    axes[0].set_ylabel("density")
    #axes[0].set_xlim([-1, 1])
    # sparse: sharp peak at 0 with heavy tails
    # dense:  broad bell curve

    # log scale to see tails
    axes[1].hist(np.abs(all_rs), bins=100, density=True,
                 color='steelblue', alpha=0.7, log=True)
    axes[1].set_title("Distribution of |rh| (log scale)")
    axes[1].set_xlabel("|rh| value")
    axes[1].set_ylabel("log density")
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    #axes[1].set_xlim([0, 1])
    # sparse: straight line on log scale = exponential tails
    # dense:  curved = Gaussian tails

    # print summary stats
    print(f"mean |rh|  = {np.mean(np.abs(all_rs)):.4f}")
    print(f"std r     = {np.std(all_rs):.4f}")
    print(f"kurtosis  = {float(np.mean((all_rs - np.mean(all_rs))**4) / np.std(all_rs)**4):.2f}")
    # kurtosis > 3 means super-Gaussian (sparse)
    # kurtosis ≈ 3 means Gaussian (not sparse)
    # kurtosis < 3 means sub-Gaussian (too uniform)
    print(f"sparsity (|r|<0.1) = {np.mean(np.abs(all_rs) < 0.1):.2f}")
    print(f"sparsity (|r|<0.5) = {np.mean(np.abs(all_rs) < 0.5):.2f}")
    print(f"percentiles: "
          f"50%={np.percentile(np.abs(all_rs), 50):.3f} "
          f"75%={np.percentile(np.abs(all_rs), 75):.3f} "
          f"90%={np.percentile(np.abs(all_rs), 90):.3f} "
          f"99%={np.percentile(np.abs(all_rs), 99):.3f}")

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.show()

    return all_rs

# inference only on models 

def plot_endstopping(model, bars, dataset, sparse=False, savefile = "figures/endstopping_dense.jpg", l2 =True):
    """
    Show input patch vs reconstruction side by side.
    If filters are learning, these should look similar.
    """
    fig, axes = plt.subplots(4,8, figsize = (25, 12))
    #handles, labels = axes.get_legend_handles_labels()
    n = len(bars)
    rs_all = []
    r_tds_all = []
    error_tds_all = []
    rh_all = []
    for idx in range(n):
        images = []
        for i in range(3):
            x = dataset.overlap * i
            # Apply gaussian mask
            image = bars[idx][:, x:x+dataset.h_pix].reshape([-1])
            if sparse:
                image = image * dataset.mask

            images.append(image)
        if sparse:
            rs, r_tds, rh, error_tds,_,_ = model.apply_images_sparse(
                images, training=False)
        else:
            rs, r_tds, rh, error_tds,_,_ = model.apply_images(
                images, training=False)
            
        rs_all.append(rs)
        r_tds_all.append(r_tds)
        error_tds_all.append(error_tds)
    
    rs_all = np.array(rs_all)
    r_tds_all = np.array(r_tds_all)
    error_tds_all = np.array(error_tds_all)
    axes = axes.flatten()
    idx_c=1
    for i in range(32):
        axes[i].plot(np.abs(rs_all[:,32*idx_c+i]), '--')
        if l2:
            axes[i].plot(np.abs(r_tds_all[:,32*idx_c+i]), '--')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if np.mod(i, 8) == 0:
            axes[i].set_ylabel("Response (A.U.)")

        if i>= 8*4:
            axes[i].set_xlabel("Bar size (px)")

        
        #axes[i].plot(rs_all[:,i]-r_tds_all[:,i], '*')

    labels = ["L1 response (r)", "Top-Down Response (r_td)"]
    plt.figlegend(labels, loc='upper right')
    plt.suptitle("Endstopping - Horizontal Bar")
    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.show()

    fig, axes = plt.subplots(len(bars),3, figsize = (10, 60)) # for each bar length, plot the r, r_td, and error
    print(rs_all.shape)
    norm_e = []
    for i in range(n):
        axes[i,0].bar(np.arange(0,32,1), rs_all[i,32*idx_c:32*(idx_c+1)], facecolor='k', width=0.5)
        axes[i,1].bar(np.arange(0,32,1), r_tds_all[i,32*idx_c:32*(idx_c+1)], facecolor='k', width=0.5)
        axes[i,2].bar(np.arange(0,32,1), error_tds_all[i,32*idx_c:32*(idx_c+1)], facecolor='k', width=0.5)
        axes[i,0].set_ylim([-2,2])
        axes[i,1].set_ylim([-2,2])
        axes[i,2].set_ylim([-2,2])
        axes[0,0].set_title("rs")
        axes[0,1].set_title("r_tds")
        axes[0,2].set_title("r - r_tds")
        axes[i,0].set_ylabel("bar length = "+str(i))
        for j in range(3):
            axes[i,j].set_xlabel("unit #")

            axes[i,j].spines['top'].set_visible(False)
            axes[i,j].spines['right'].set_visible(False)

        norm_e.append(np.linalg.norm(error_tds_all[i,32*idx_c:32*(idx_c+1)]))


    plt.savefig(savefile[:-4]+"_responses.jpg")
    plt.show()

    print(f"Min norm err:{np.argmin(np.array(norm_e))}")
    print(f"Max norm err:{np.argmax(np.array(norm_e))}")

    return error_tds_all[:, 32:64]


def plot_power_spectrum(images, title="Power spectrum"):
    """
    Radially averaged power spectrum for images of different sizes.
    Normalises each image's PSD to unit mean before averaging,
    so different sized images contribute equally.
    """
    all_psds  = []
    all_freqs = []

    for img in images:
        img = np.array(img, dtype=np.float64)
        h, w = img.shape

        # 2D FFT
        f     = np.fft.fft2(img)
        psd2d = np.abs(np.fft.fftshift(f))**2

        # radial frequency grid — normalise by image size
        # so frequencies are in cycles/pixel (0 to 0.5)
        # this makes PSDs from different sized images comparable
        cy, cx = h//2, w//2
        Y, X   = np.mgrid[-cy:h-cy, -cx:w-cx]
        R      = np.sqrt((X/w)**2 + (Y/h)**2)   # normalised frequency

        # bin into 100 frequency bins from 0 to 0.5
        n_bins  = 100
        f_bins  = np.linspace(0, 0.5, n_bins + 1)
        f_cents = 0.5 * (f_bins[:-1] + f_bins[1:])   # bin centres
        psd1d   = np.zeros(n_bins)
        counts  = np.zeros(n_bins)

        R_flat   = R.ravel()
        psd_flat = psd2d.ravel()

        for k in range(n_bins):
            mask        = (R_flat >= f_bins[k]) & (R_flat < f_bins[k+1])
            if mask.sum() > 0:
                psd1d[k]   = psd_flat[mask].mean()
                counts[k]  = mask.sum()

        # normalise each image's PSD to unit mean
        # so large and small images contribute equally
        valid = psd1d > 0
        if valid.sum() > 0:
            psd1d /= psd1d[valid].mean()
            all_psds.append(psd1d)
            all_freqs.append(f_cents)

    # average across all images
    avg_psd = np.mean(all_psds, axis=0)
    freqs   = all_freqs[0]   # same bins for all

    # only fit over valid non-zero bins, skip DC (bin 0)
    valid   = (avg_psd > 0) & (freqs > 0)
    log_f   = np.log(freqs[valid])
    log_p   = np.log(avg_psd[valid])
    slope, intercept = np.polyfit(log_f, log_p, 1)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # linear scale
    axes[0].plot(freqs[valid], avg_psd[valid], 'b-', linewidth=1.5)
    axes[0].set_xlabel("Normalised spatial frequency (cycles/pixel)")
    axes[0].set_ylabel("Power")
    
    axes[0].set_title(f"{title} — linear scale")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # log-log scale with fit
    axes[1].loglog(freqs[valid], avg_psd[valid], 'b-', linewidth=1.5,
                   label="observed")
    axes[1].loglog(freqs[valid],
                   np.exp(intercept) * freqs[valid]**slope,
                   'r--', linewidth=1.5,
                   label=f"fit: slope={slope:.2f}")
    axes[1].set_xlabel("Normalised spatial frequency (log)")
    axes[1].set_ylabel("Power (log)")
    axes[1].set_title(f"{title} — log-log scale")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend()

    plt.suptitle(title,
        fontsize=11)
    plt.tight_layout()
    #plt.savefig(f"psd_{title.replace(' ','_')}.png", dpi=150)
    plt.show()

    print(f"[{title}]")
    print(f"  slope        = {slope:.3f}")
    print(f"  n images     = {len(images)}")
    print(f"  interpretation:")
    if slope < -1.5:
        print(f"  → natural / correlated  (expected ≈ -2.0)")
    elif -1.5 <= slope < -0.5:
        print(f"  → partially whitened    (DoG range)")
    elif -0.5 <= slope <= 0.5:
        print(f"  → well whitened         (target for this model)")
    else:
        print(f"  → over-whitened / high freq dominated")

    return freqs, avg_psd, slope


def get_bar_patch(bar_width, patches):
    """
    Get bar patch image for end stopping test.
    """
    bar_patch = np.ones((16,26), dtype=np.float32)

    
    bar_height = 2

    for x in range(bar_patch.shape[1]):
        for y in range(bar_patch.shape[0]):
            if x >= 26/2 - bar_width/2 and \
            x < 26/2 + bar_width/2 and \
            y >= 16/2 - bar_height/2 and \
            y < 16/2 + bar_height/2:
                bar_patch[y,x] = -1.0

            

    # Sete scale with stddev of all patch images.
    scale = np.std(patches)
    bar_patch = (bar_patch - np.mean(bar_patch))/(np.std(bar_patch)+1e-6)
    
    # Original scaling value for bar
    #bar_scale = 2.0
    return bar_patch #* scale #* bar_scale

def calc_errneur_endstopping(err_neurs):
    # for each neur, calculate the diff between peak and flat
    ESI = np.zeros(err_neurs.shape[1])
    for i in range(len(err_neurs)):
        r = np.abs(err_neurs[:,i])
        peak = np.max(r[0:18])
        ESI = (peak - r)/(peak + 1e-6)

    return ESI
