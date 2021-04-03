import copy

import numpy as np

def cube_sphere_mesh(dim):
    """
    Generates a 6 part cubic sphere mesh
    
    Parameters
    ----------
    dim: int
        The side length of each face in the mesh

    Returns
    -------
    faces: array
        Array of faces containing the x, y and z mesh co-ordinates
    """
    # Generate points on one face of a cube
    lin = np.linspace(-1, 1, dim, endpoint=True)
    ax2, ax1 = np.meshgrid(lin, lin)
    ax1 *= -1
    ax3 = np.ones([dim, dim])

    # Map cube face outwards onto sphere
    r = np.sqrt(ax1**2 + ax2**2 + ax3**2)
    ax1 /= r
    ax2 /= r
    ax3 /= r

    # Return list of 6 faces
    faces = []
    faces.append([copy.copy(ax3), copy.copy(ax2), copy.copy(ax1)])
    faces.append([copy.copy(-ax2), copy.copy(ax3), copy.copy(ax1)])
    faces.append([copy.copy(-ax3), copy.copy(-ax2), copy.copy(ax1)])
    faces.append([copy.copy(ax2), copy.copy(-ax3), copy.copy(ax1)])
    faces.append([copy.copy(-ax1), copy.copy(ax2), copy.copy(ax3)])
    faces.append([copy.copy(ax1), copy.copy(ax2), copy.copy(-ax3)])

    return np.array(faces)
