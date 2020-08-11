import numpy as np
import cv2
import random
from typing import Tuple


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

    def __init__(self, target_image: np.ndarray, genes: int = 128, genotype=[]):
        """
        Attributes:
            target_image(np.ndarray): The target image the class is to be trained on
            genes(Int): The number of circles used to create a trained image
            mutation_rate(float): The chance of a gene being altered on the creation of a new generation
        """
        self.target_image: np.ndarray = target_image
        self.shape: Tuple[int, int] = target_image.shape
        self.genes: int = genes

        # initial value a placeholder/ blank black canvas
        self.phenotype: np.ndarray = np.zeros(
            target_image.shape, dtype="uint8")

        self.average_radius: float = (self.shape[1] + self.shape[0])/2 / 6

        self.max_error: float = (np.square((1 - (self.target_image >= 127))
                                           * 255 - self.target_image)).mean(axis=None)
        if (genotype == []):
            self.genotype: np.ndarray = np.random.rand(self.genes, 5)
        else:
            self.genotype: np.ndarray = genotype

    def gen_phenotype(self) -> None:
        """ Creates an image using the genotype and overwrites self.phenotype
        """

        # clear any previous image
        self.phenotype.fill(0)

        for gene in self.genotype:
            overlay = self.phenotype.copy()
            color = (int(gene[3] * 255),)
            cv2.circle(
                overlay,
                center=(int(gene[1] * self.shape[1]),
                        int(gene[0]*self.shape[0])),
                radius=int(gene[2] * self.average_radius),
                color=color,
                thickness=-1

            )

            alpha = gene[-1]
            self.phenotype = cv2.addWeighted(
                overlay, alpha, self.phenotype, 1 - alpha, 0).astype(int)

    def fitness(self) -> float:
        """ Sets the fitness score of the specie based on current phenotype.
        Scored based on Mean Squared Error"""
        cost: int = int(np.mean(np.square(self.target_image - self.phenotype)))
        fitness = (self.max_error - cost) / self.max_error
        return fitness
