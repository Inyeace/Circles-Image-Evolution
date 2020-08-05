import numpy as np


class Specie:
    """ Class that holds information on the trained images
        Attributes:
            target_image (np.ndarray): Target image to train class on.
            shape(tuple): Shape of the target image and the trained image. 
            circles(Int): The number of circles used to create a trained image.
            mutation_rate(float): The chance of a gene being altered on the creation of a new generation.
            genotype(np.ndarray): 2D array of size self.circles where each element array holds the information of 
                circle center postion, radius, greyscale value and alpha value.


     """

    def __init__(self, target_image, genes=128, mutation_rate=0.01):
        """
        Attributes:
            target_image(np.ndarray): The target image the class is to be trained on
            genes(Int): The number of circles used to create a trained image
            mutation_rate(float): The chance of a gene being altered on the creation of a new generation
        """
        self.target_image = target_image
        self.shape = target_image.shape
        self.genes = genes
        self.mutation_rate = mutation_rate
        self.genotype = np.random.rand(genes,5)

        #initial value a placeholder/ blank black canvas
        self.phenotype = np.zeros(target_image.shape)
        

