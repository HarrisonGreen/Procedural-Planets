import numpy as np

def random(sea_level):
    water = np.random.randint(0, 256, 3)
    lowland = np.random.randint(0, 256, 3)
    highland = np.random.randint(0, 256, 3)

    water = f'rgb{water[0], water[1], water[2]}'
    lowland = f'rgb{lowland[0], lowland[1], lowland[2]}'
    highland = f'rgb{highland[0], highland[1], highland[2]}'

    return [[0, water], [sea_level, lowland], [1, highland]]
