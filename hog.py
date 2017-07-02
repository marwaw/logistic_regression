import numpy as np

def extract(img):
    """
    :param img: image, nparray shape(3136,)
    :return: extracted features, nparray shape(175, )
    """
    img = np.reshape(img, (56,56))
    result = hog(img)
    return result

def extract_all(x):
    """
    :param x: macierz obrazów do ekstrakcji, nparray shape(N,3136)
    :return: extracted features, nparray shape(N,175)
    """
    return np.array(list(map(extract, x)))

def hog(image):
    nwin_x = 5
    nwin_y = 5
    B = 7
    (L,C) = np.shape(image)
    H = np.zeros(shape=(nwin_x*nwin_y*B,1))
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C/(nwin_x+1))
    step_y = np.floor(L/(nwin_y+1))
    cont = 0
    hx = np.array([[1,0,-1]])
    hy = np.array([[-1],[0],[1]])
    grad_xr = convolve(image, hx)
    grad_yu = convolve(image, hy)
    angles = np.arctan2(grad_yu,grad_xr)
    magnit = np.sqrt((grad_yu**2 +grad_xr**2))
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            magnit2 = magnit[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            K = np.shape(v_angles)[0]
            bin = 0
            H2 = np.zeros(shape=(B,1))
            for ang_lim in np.arange(start=-np.pi+2*np.pi/B,stop=np.pi+2*np.pi/B,step=2*np.pi/B):
                for k in range(K):
                    if v_angles[k]<ang_lim:
                        v_angles[k]=100
                        H2[bin]+=v_magnit[k]
                bin += 1

            H2 = H2 / (np.linalg.norm(H2)+0.01)
            H[(cont-1)*B:cont*B]=H2

    return H.reshape(175)

def convolve(image, h):
    wynik = np.zeros(shape=np.shape(image))
    if (h.shape[1] == 3): # hx
        for i in range(image.shape[0]):
            x = image[i,:]
            wynik[i, :] = np.convolve(x, h[0,:], mode='same')
    else:
        for i in range(image.shape[1]):
            y = image[:, i]
            wynik[:, i] = np.convolve(y , h[:,0], mode='same')
    return wynik
