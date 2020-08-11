from species import Specie
from typing import List
import numpy as np
import random
import cv2
import os

class Evolution:
    def __init__(self, targetImage: np.ndarray,
                 genes: int = 128):
        self.targetImage = targetImage
        self.genes = genes
        self.specie = Specie(targetImage, genes)

    def mutate(self, specie: Specie) -> Specie:
        new_specie = Specie(self.targetImage, self.genes,
                            genotype=np.array(specie.genotype))
        # Randomization for Evolution

        y = random.randint(0, self.genes - 1)

        # random number of floats to change in a gene
        change = random.randint(0, new_specie.genotype.shape[1] + 1)

        # random set of floats to change in a gene by their index 
        selection = np.random.choice(
            new_specie.genotype.shape[1], size=change)

        if random.random() < 0.25:
            new_specie.genotype[y, selection] = np.random.rand(len(selection)) 
        else:
            new_specie.genotype[y,
                                selection] += (np.random.rand(len(selection)) - 0.5) / 3
            new_specie.genotype[y, selection] = np.clip(
                new_specie.genotype[y, selection], 0, 1)

        return new_specie

    def evolve(self, generations: int = 5000):
        self.specie.gen_phenotype()
        fitness = self.specie.fitness()


        for gen in range(generations):
            mutated = self.mutate(self.specie)
            mutated.gen_phenotype()
            newFit = mutated.fitness()
            if newFit > fitness:
                fitness = newFit
                self.specie = mutated

            print(f'Current Gen: {gen + 1} Fitness: {fitness}')
            os.system("cls")
            
            cv2.imshow("Evolving Circles", cv2.convertScaleAbs(self.specie.phenotype))
            cv2.waitKey(1)
        cv2.imshow("Evolving Circles",
                   cv2.convertScaleAbs(self.specie.phenotype))
        cv2.waitKey(0)


        
