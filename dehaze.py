from PIL import Image, ImageFilter, ImageDraw
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import math

np.seterr(divide='ignore', invalid='ignore')


def read_image(img_path):
    im = Image.open(img_path).convert("RGB")
    imdata = np.array(im, dtype=float) / 255.
    imdata = white_balance(imdata)
    gray_im = imdata[:, :, 0] * 299. / 1000. + imdata[:, :, 1] * 587. / 1000. + imdata[:, :, 2] * 114. / 1000.
    return imdata, gray_im


def estimate_airlight(rgb, gray):
    block_M = 30
    h, w = gray.shape
    block_gray = np.zeros(gray.shape, dtype=float)
    for i in range(0, h, block_M):
        for j in range(0, w, block_M):
            block_gray[i: i + block_M, j: j + block_M] = np.min(gray[i: i + block_M, j: j + block_M])

    start_i, start_j = 0, 0
    end_i, end_j = h, w
    half_i, half_j = int((end_i - start_i) / 2), int((end_j - start_j) / 2)
    block_th = min(half_i, half_j)
    while block_M < block_th:
        ave_block = np.zeros((2, 2), dtype=float)
        for i in range(2):  # start_i, end_i, half_i):
            for j in range(2):  # (start_j, end_j, half_j):
                ave_block[i, j] = np.average(block_gray[i: i + half_i, j: j + half_j])
        max_i, max_j = np.argwhere(ave_block == np.max(ave_block))[0]
        start_i, start_j = start_i + max_i * half_i, start_j + max_j * half_j
        end_i, end_j = start_i + half_i, start_j + half_j
        half_i, half_j = int((end_i - start_i) / 2), int((end_j - start_j) / 2)
        block_th = min(half_i, half_j)
    euclidean_norm = np.sqrt(np.sum((rgb[start_i: end_i, start_j: end_j] - 1) ** 2, axis=2))
    pi, pj = np.argwhere(euclidean_norm == np.min(euclidean_norm))[0]
    A = rgb[pi + start_i, pj + start_j]
    return A


def estimate_transmission(A, rgb, gray, patch_ize=15, w=0.95):
    # Calculations are based on the
    # https: // link.springer.com / article / 10.1186 / s13640 - 016 - 0104 - y
    pad = int(patch_ize / 2)
    t_bar = np.zeros(rgb.shape[0:2], dtype=float)
    I_over_A = rgb / A
    IA_min = np.pad(np.min(I_over_A, axis=2), (pad, pad), 'constant', constant_values=(1., 1.))
    for i in range(pad, IA_min.shape[0] - pad):
        for j in range(pad, IA_min.shape[1] - pad):
            t_bar[i - pad, j - pad] = 1 - w * np.min(IA_min[i - pad: i + pad + 1, j - pad: j + pad + 1])
    guide_t_bar = guided_filter(rgb, t_bar, 20, 0.001)
    return guide_t_bar


def guided_filter(guide_image, filtering_image, r, eps):
    if len(filtering_image.shape) == 2:
        filtering_image = filtering_image[..., np.newaxis]
    if len(guide_image.shape) == 2:
        guide_image = guide_image[..., np.newaxis]
    filterN = box_filter(np.ones(guide_image.shape[0:2]), r)

    meanI = box_filter(guide_image, r) / filterN
    meanP = box_filter(filtering_image, r) / filterN
    meanIP = box_filter(guide_image * filtering_image, r) / filterN
    covIP = meanIP - meanI * meanP
    meanII = box_filter(filtering_image * filtering_image, r) / filterN
    varI = meanII - meanI * meanI

    ax = covIP / (varI + eps)
    bx = meanP - ax * meanI

    meanAx = box_filter(ax, r) / filterN
    meanBx = box_filter(bx, r) / filterN

    guided_q = meanAx * guide_image + meanBx

    return guided_q


def box_filter(inImg, fsize):
    if len(inImg.shape) == 2:
        inImg = inImg[..., np.newaxis]
    rw, cl, bd = inImg.shape
    outImg = np.zeros_like(inImg)

    # Cumulative Sum along the rows
    imgCumSum = np.cumsum(inImg, 0)
    outImg[0: fsize + 1, :, :] = imgCumSum[fsize: 2 * fsize + 1, :, :]
    outImg[fsize + 1: rw - fsize, :, :] = imgCumSum[2 * fsize + 1: rw, :, :] - imgCumSum[0: rw - 2 * fsize - 1, :, :]
    outImg[rw - fsize: rw, :, :] = np.tile(imgCumSum[rw - 1, :, :], [fsize, 1, 1]) \
                                   - imgCumSum[rw - 2 * fsize - 1: rw - fsize - 1, :, :]

    # Cumulative Sum along the columns
    imgCumSum = np.cumsum(outImg, 1)
    outImg[:, 0: fsize + 1, :] = imgCumSum[:, fsize: 2 * fsize + 1]
    outImg[:, fsize + 1: cl - fsize, :] = imgCumSum[:, 2 * fsize + 1: cl, :] - imgCumSum[:, 0: cl - 2 * fsize - 1, :]
    outImg[:, cl - fsize: cl, :] = np.tile(imgCumSum[:, cl - 1, :], [fsize, 1, 1]).transpose(1, 0, 2) \
                                   - imgCumSum[:, cl - 2 * fsize - 1: cl - fsize - 1, :]
    return outImg


def dehaze_image(imgdata, A, T):
    t0 = 0.1
    T[T < t0] = t0

    if len(T.shape) == 2:
        T = T[..., np.newaxis]

    J = (imgdata - A) / T + A
    J[J < 0] = 0
    J[J > 1] = 1
    print(np.min(J), np.max(J))
    return J


def white_balance(channels, percentage=0.05):
    if len(channels.shape) == 2:
        print("Please use 3-band image for white balance. ")
    elif len(channels.shape) == 3:
        if channels.shape[2] == 1:
            print("Please use 3-band image for white balance.")
        else:
            balanced = np.zeros_like(channels)
            for b in range(balanced.shape[2]):
                min_pix, max_pix = np.percentile(channels[:, :, b], percentage), np.percentile(channels[:, :, b], 100. - percentage)
                balanced[:, :, b] = np.clip((channels[:, :, b] - min_pix) / (max_pix - min_pix), 0., 1.)
            return balanced


def estimate_transmission2(A, rgb, gray):
    # Calculations are based on
    # https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569911081.pdf
    # Calculate entropy
    r = np.arange(0, 1.02, 0.02)  # Transmission, first value s 0.01 to avoid divide by zero
    r[0] = 0.01
    transmittance = np.zeros_like(gray)
    block = 30
    height, width = gray.shape
    N = block * block  # Total number of pixes in a block
    AL = A[0] * 299. / 1000. + A[1] * 587. / 1000. + A[2] * 114. / 1000.

    for i in range(0, height, block):
        start_i, end_i = i, i + block
        if end_i > height: end_i = height
        for j in range(0, width, block):
            start_j, end_j = j, j + block
            if end_j > width: end_j = width
            transmittance[start_i: end_i, start_j: end_j] = \
                calculate_transmittance(gray[start_i: end_i, start_j: end_j], AL, r)


def calculate_transmittance(gr, atm, rad):
    entropy = np.zeros_like(rad)
    fidelity = np.zeros_like(rad)
    for i in range(len(rad)):
        J = ((gr - atm) / rad[i] + atm) * 255
        entropy[i] = calculate_entropy(J, atm, rad[i])
        fidelity[i] = calculate_fideity(J)
    objective = entropy * fidelity
    print('Entropy:', entropy)
    print('Fidelity: ', fidelity)
    plt.plot(r, entropy, '-o')
    plt.plot(r, fidelity, '-o')
    plt.plot(r, objective, '-o')
    plt.show()


def calculate_fideity(img):
    N = img.shape[0] * img.shape[1]
    img[img > 255] = 0
    img[img < 0] = 0
    img[img > 0] = 1
    delta_p = np.zeros((3,), dtype=float)
    for b in range(img.shape[2]):
        delta_p[b] = np.sum(img[:, :, b])
    sr = delta_p / N
    fidel = np.min(sr)
    return fidel


def calculate_entropy(img_J, N):
    h = np.zeros((256,), dtype=float)
    N = img_J.shape[0] * img_J.shape[1]
    for i in range(img_J.shape[0]):
        for j in range(img_J.shape[1]):
            if 0 <= img_J[i, j] <= 255:
                h[int(img_J[i, j])] += 1
    hn = h / N
    entrp = 0.
    for el in hn:
        if el != 0:
            entrp += el * np.log(el)
    return -entrp


def display_image(inarray2D):
    disp = np.zeros((inarray2D.shape[0], inarray2D.shape[1], 3), dtype=np.uint8)
    if len(inarray2D.shape) == 2:
        disp[:, :, 0] = np.uint8(inarray2D * 255)
        disp[:, :, 1] = np.uint8(inarray2D * 255)
        disp[:, :, 2] = np.uint8(inarray2D * 255)
    elif len(inarray2D.shape) == 3 and inarray2D.shape[2] == 1:
        disp[:, :, 0] = np.uint8(inarray2D[:, :, 0] * 255)
        disp[:, :, 1] = np.uint8(inarray2D[:, :, 0] * 255)
        disp[:, :, 2] = np.uint8(inarray2D[:, :, 0] * 255)
    else:
        disp = np.uint8(inarray2D * 255)
    dispimg = Image.fromarray(disp, 'RGB')
    dispimg.show()


if __name__ == '__main__':
    path = "toJasper.jpg"
    img, g_img = read_image(path)
    atm = estimate_airlight(img, g_img)
    # chroma = chromaticity(img)
    # haze = haze_image(img, chroma)
    # t = estimate_transmission(a, img, haze) #radiance(img, haze, g_img)
    # dehaze = dehaze_image(img, a, t)
    # dispImg = Image.fromarray(np.uint8(dehaze * 255))
    # dispImg.save('outputImage.jpg')
    # dispImg.show()
    # estimate_transmission2(a, img, g_img)
    trns = estimate_transmission(atm, img, g_img)
    dehaze = dehaze_image(img, atm, trns)
    display_image(dehaze)



'''
def chromaticity(imdata):
    sigma = np.zeros(imdata.shape, dtype=float)
    for b in range(imdata.shape[2]):
        sigma[:, :, b] = np.nan_to_num(imdata[:, :, b] / np.sum(imdata, axis=2))
    return sigma


def haze_image(imdata, sigma):
    sigma_min = np.min(sigma, axis=2)
    lmbda = np.zeros(imdata.shape, dtype=float)
    for b in range(imdata.shape[2]):
        lmbda[:, :, b] = np.nan_to_num((sigma[:, :, b] - sigma_min) / (1 - 3 * sigma_min))
    lmbda_max = np.max(lmbda, axis=2)
    haze = (np.max(imdata, axis=2) - lmbda_max * np.sum(imdata, axis=2)) / (1 - 3 * lmbda_max)
    haze = gaussian_filter(haze, sigma=1)
    return haze


def radiance(imdata, hazedata, gdata):
    t0 = 0.1
    num_top_pixels = int(0.001 * hazedata.shape[0] * hazedata.shape[1]) - 1
    haze1d = hazedata.flatten()
    haze1d = np.sort(haze1d)[::-1]
    th_pixel = haze1d[num_top_pixels]
    top_haze_pixels = (hazedata >= th_pixel) * 1.
    radiance_a = gdata * top_haze_pixels
    max_indx = np.where(radiance_a == np.amax(radiance_a))
    max_r_c = list(zip(max_indx[0], max_indx[1]))[0]
    A = imdata[max_r_c]

    k0 = 1 - np.min(imdata, axis=2)
    k1 = hazedata - np.min(imdata, axis=2)

    transmission_t = np.nan_to_num((hazedata[max_r_c] * k0 - hazedata * k0) / (hazedata[max_r_c] * k0 - hazedata * k1))
    transmission_t[transmission_t < t0] = t0
    dehaze = np.zeros(imdata.shape, dtype=float)
    for b in range(imdata.shape[2]):
        dehaze[:, :, b] = (imdata[:, :, b] - A[b]) / transmission_t + A[b]
    print(np.min(dehaze), np.max(dehaze))

    return dehaze


def gaussian_kernel2D(size, sigma=1):
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    krnl = np.zeros((size, size), np.float)
    for i in range(size):
        for j in range(size):
            krnl[i, j] = 1 / np.sqrt(2 * np.pi * sigma ** 2) \
                         * np.exp(-(x[i] ** 2 + y[j] ** 2) / (2 * sigma ** 2))
    return krnl / np.sum(krnl)
    

def estimate_transmission2(A, rgb, gray):
    # Calculations are based on
    # https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569911081.pdf
    # Calculate entropy
    r = np.arange(0, 1.025, 0.025)  # Transmission, first value s 0.01 to avoid divide by zero
    r[0] = 0.01
    N = gray.shape[0] * gray.shape[1]  # Total number of pixes
    AL = A[0] * 299 / 1000 + A[1] * 587 / 1000 + A[2] * 114 / 1000
    entropy = np.zeros(r.shape, dtype=float)
    fidelity = np.zeros(r.shape, dtype=float)
    for i in range(len(r)):
        J = ((img - A) / r[i] + A) * 255
        entropy[i] = calculate_entropy(gray, AL, r[i])
        fidelity[i] = calculate_fideity(J)
    objective = entropy * fidelity
    print('Entropy:', entropy)
    print('Fidelity: ', fidelity)
    plt.plot(r, entropy, '-o')
    plt.plot(r, fidelity, '-o')
    plt.plot(r, objective, '-o')
    plt.show()


def calculate_fideity(img):
    N = img.shape[0] * img.shape[1]
    img[img > 255] = 0
    img[img < 0] = 0
    img[img > 0] = 1
    delta_p = np.zeros((3,), dtype=float)
    for b in range(img.shape[2]):
        delta_p[b] = np.sum(img[:, :, b])
    sr = delta_p / N
    fidel = np.min(sr)
    return fidel


def calculate_entropy(img_g, atm, r):
    J_gray = ((img_g - atm) / r + atm) * 255
    h = np.zeros((256,), dtype=float)
    N = img_g.shape[0] * img_g.shape[1]
    for i in range(img_g.shape[0]):
        for j in range(img_g.shape[1]):
            if 0 <= J_gray[i, j] <= 255:
                h[int(J_gray[i, j])] += 1
    hn = h / N
    entrp = 0.
    for el in hn:
        if el != 0:
            entrp += el * np.log(el)
    return -entrp


def estimate_transmission(A, imdata, hazedata):
    minI = np.zeros((3,), dtype=float)
    trns = np.zeros(imdata.shape, dtype=float)
    for b in range(imdata.shape[2]):
        minI[b] = np.min(imdata[:, :, b])
    k0 = 1 - minI

    for b in range(3):
        k1 = hazedata - minI[b]
        trns[:, :, b] = (A[b] * k0[b] - hazedata * k0[b]) / (A[b] * k0[b] - hazedata * k1)
    return trns
'''
