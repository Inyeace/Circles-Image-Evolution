from evolution import Evolution
from helpers import load_target_image
from species import Specie
import numpy as np
import cv2
target_image = load_target_image('Mona Lisa 128.jpg')




evo = Evolution(target_image,genes=256)
evo.evolve(100)
print(target_image)
print(cv2.convertScaleAbs(evo.specie.phenotype))
print(target_image.shape == evo.specie.phenotype.shape)





