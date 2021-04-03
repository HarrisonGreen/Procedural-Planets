import numpy as np

def composite_perlin(components, octaves, base_freq, freq_ratio, decay, damp, power):
    """
    Generates Perlin noise with the given parameters for a list of components

    Parameters
    ----------
    components: list
        A list, whose elements should each contain 3 matrices with x, y and z co-ordinates (i.e. mesh
        components of a surface)
    octaves: int
        Number of layers of noise to compute (usually around 4 or 5)
    base_freq: int
        Length of gradient vector field used in the first layer
    freq_ratio: int
        Ratio of vector frequency between one layer and the next (usually 2)
    decay: float
        Ratio of amplitude between one layer and the next (usually around 0.5)
    damp: float
        Final damping factor for the noise
    power: float
        Power used to manipulate noise shape

    Returns
    -------
    components: list
        The updated components with noise included
    composite_noise: list
        The new heights of the points; used for colouring
    """
    # Get the required number of layers of noise
    noise_layers = []
    for i in range(octaves):
        noise_layers.append(perlin_noise(components, base_freq * freq_ratio ** i) * decay ** i)

    # Return the sum of the different layers
    noise_layers = np.array(noise_layers)
    composite_noise = [sum(noise_layers[:, i]) for i in range(len(noise_layers[0]))]
    for i in range(len(composite_noise)):
        composite_noise[i] = 1 + (np.sign(composite_noise[i]) * abs(composite_noise[i]) ** power)/damp

    return composite_noise

def perlin_noise(components, vecs):
    "Generates noise array for each component using the given number of gradient vectors"
    # Create random permutation to use for determining gradient vectors
    permutation = np.arange(256, dtype = int)
    np.random.shuffle(permutation)
    shuffle = np.stack([permutation, permutation]).flatten()

    # Rescale co-ordinates and get noise for each component
    step = 2/(vecs - 2)
    noise_components = []
    for component in components:
        x = (component[0] + 1 + step/2)/step
        y = (component[1] + 1 + step/2)/step
        z = (component[2] + 1 + step/2)/step

        noise_components.append(perlin_noise_component(x, y, z, shuffle))

    return np.array(noise_components)

def perlin_noise_component(x, y, z, shuffle):
    "Generates array of noise at given co-ordinates with specified shuffle"
    # Split co-ordinates into integer and fractional parts
    xi = x.astype(int)
    yi = y.astype(int)
    zi = z.astype(int)
    xf = x - xi
    yf = y - yi
    zf = z - zi

    # Calculate dot products of position vectors with 8 surrounding gradient vectors
    n000 = dot_prod(shuffle[shuffle[shuffle[xi] + yi] + zi], xf, yf, zf)
    n001 = dot_prod(shuffle[shuffle[shuffle[xi] + yi] + zi + 1], xf, yf, zf - 1)
    n010 = dot_prod(shuffle[shuffle[shuffle[xi] + yi + 1] + zi], xf, yf - 1, zf)
    n100 = dot_prod(shuffle[shuffle[shuffle[xi + 1] + yi] + zi], xf - 1, yf, zf)
    n011 = dot_prod(shuffle[shuffle[shuffle[xi] + yi + 1] + zi + 1], xf, yf - 1, zf - 1)
    n101 = dot_prod(shuffle[shuffle[shuffle[xi + 1] + yi] + zi + 1], xf - 1, yf, zf - 1)
    n110 = dot_prod(shuffle[shuffle[shuffle[xi + 1] + yi + 1] + zi], xf - 1, yf - 1, zf)
    n111 = dot_prod(shuffle[shuffle[shuffle[xi + 1] + yi + 1] + zi + 1], xf - 1, yf - 1, zf - 1)

    # Calculate smoothed co-ordinate values
    u = smooth(xf)
    v = smooth(yf)
    w = smooth(zf)

    # Interpolate between dot products
    x1 = interp(n000, n100, u)
    x2 = interp(n001, n101, u)
    x3 = interp(n010, n110, u)
    x4 = interp(n011, n111, u)
    y1 = interp(x1, x3, v)
    y2 = interp(x2, x4, v)
    z1 = interp(y1, y2, w)
    return z1

def dot_prod(i, x, y, z):
    "Converts index i into gradient vectors and returns dot product with (x, y, z)"
    vectors = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
    grad = vectors[i%6]
    return grad[:, :, 0] * x + grad[:, :, 1] * y + grad[:, :, 2] * z

def interp(a, b, x):
    "Linear interpolation between values a and b with weight x"
    return a + x * (b-a)

def smooth(t):
    "Smooths values with sigmoid function '6t^5 - 15t^4 + 10t^3'"
    return 6 * t**5 - 15 * t**4 + 10 * t**3
