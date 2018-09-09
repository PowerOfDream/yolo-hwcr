import numpy as np
from PIL import Image
from PIL import ImageFilter

def gen_color_matrix(Hmean, Smean, Vmean, variance, size):
    '''
    generate a random color matrix

    Hmean: int, mean of Hue
    Smean: int, mean of Saturation
    Vmean: int, mean of Value
    variance: int, variance
    size: int tuple, (width, height)
    '''
    H = np.random.randn(size[0], size[1]) * variance + Hmean
    S = np.random.randn(size[0], size[1]) * variance + Smean
    V = np.random.randn(size[0], size[1]) * variance + Vmean

    H[H > 255.0] = 255.0
    H[H < 0.0] = 0.0
    S[S > 255.0] = 255.0
    S[S < 0.0] = 0.0
    V[V > 255.0] = 255.0
    V[V < 0.0] = 0.0

    IH = Image.fromarray(H.astype(np.uint8), 'L')
    IS = Image.fromarray(S.astype(np.uint8), 'L')
    IV = Image.fromarray(V.astype(np.uint8), 'L')

    img = Image.merge("HSV", (IH, IS, IV)).convert("RGB")
    #img = img.filter(ImageFilter.GaussianBlur(radius=2))

    return img
    #img.save("d://back.png")


def gen_bg_img(Hrange, Srange, Vrange, varRange, size):
    '''
    generate a background image by calling en_color_matrix

    Hrange: tuple (0, 256)
    Srange: tuple (0, 128)
    Vrange: tuple (127, 256)
    varRange: tuple (2, 15)
    '''
    Hmean    = np.random.randint(low = Hrange[0], high = Hrange[1])
    Smean    = np.random.randint(low = Srange[0], high = Srange[1])
    Vmean    = np.random.randint(low = Vrange[0], high = Vrange[1])
    variance = np.random.randint(low=varRange[0], high = varRange[1])

    return (Hmean, Smean, Vmean, variance), gen_color_matrix(Hmean, Smean, Vmean, variance, size)

