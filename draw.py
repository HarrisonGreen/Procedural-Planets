import numpy as np
import plotly.graph_objects as go

import cmap
from mesh import cube_sphere_mesh
from perlin_noise import composite_perlin

def get_params():
    # mesh parameters
    mesh_dim = 600 # Decrease this if it is taking too long

    # noise generation parameters
    octaves = 6 # Too large will give errors
    base_freq = 5 # Too large will give errors
    freq_ratio = 2
    decay = 0.4

    # noise scaling parameters
    damp = 8
    power = 1

    # ocean parameters
    sea_level = 0.55
    sea_fade = 20

    return mesh_dim, octaves, base_freq, freq_ratio, decay, damp, power, sea_level, sea_fade

def generate_planet(mesh_dim, octaves, base_freq, freq_ratio, decay, damp, power, sea_fade):
    # Create planet mesh and add noise
    flat_mesh_faces = cube_sphere_mesh(mesh_dim)
    noise_faces  = composite_perlin(flat_mesh_faces, octaves, base_freq, freq_ratio, decay, damp, power)
    mesh_faces = [flat_mesh_faces[i] * noise_faces[i] for i in range(len(noise_faces))]

    # Normalize and modify colour array
    min_h = np.min(noise_faces)
    max_h = np.max(noise_faces)
    colour = (noise_faces - min_h)/(max_h - min_h)
    colour = np.minimum(colour, sea_level/(np.exp(sea_fade * sea_level) - 1) * (np.exp(sea_fade * colour) - 1))

    # Create sea surface for plotting
    sea_faces = flat_mesh_faces * (min_h + sea_level * (max_h - min_h))

    return mesh_faces, colour, sea_faces

def plotly_draw(mesh_faces, colour, sea_faces, sea_level):
    # Choose colour map (curl, ice, purpor_r, icefire and electric are all good)
    # Can also choose cmap.random for a random one or any other plotly colorscale
    cs = "electric"

    # Plot planet
    fig = go.Figure()
    fig.update_layout(scene = {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False},
                               "bgcolor": "rgb(20,20,20)"})

    for i in range(len(mesh_faces)):
        fig.add_trace(go.Surface(x = mesh_faces[i][0], y = mesh_faces[i][1], z = mesh_faces[i][2],
                                 surfacecolor = colour[i], cmin = 0, cmax = 1, colorscale = cs,
                                 lighting = {"roughness": 0.3}))
        fig.add_trace(go.Surface(x = sea_faces[i][0], y = sea_faces[i][1], z = sea_faces[i][2],
                                 surfacecolor = colour[i], cmin = 0, cmax = 1, colorscale = cs,
                                 lighting = {"roughness": 0.1}))
    fig.show()

if __name__ == "__main__":
    mesh_dim, octaves, base_freq, freq_ratio, decay, damp, power, sea_level, sea_fade = get_params()
    mesh_faces, colour, sea_faces = generate_planet(mesh_dim, octaves, base_freq, freq_ratio, decay, damp, power, sea_fade)
    plotly_draw(mesh_faces, colour, sea_faces, sea_level)
