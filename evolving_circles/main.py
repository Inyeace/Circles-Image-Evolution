"""CLI for the app"""

import argparse
from helpers import load_target_image, save_image
from evolution import Evolution

def main():
    parser = argparse.ArgumentParser(description="Evolutionary Art Using Circles")
    parser.add_argument("image",type=str,help="Image to be processed, by default refers to images in the input folder")
    parser.add_argument("--genes",type=int,help="Number of genes/ circles", default=128)
    parser.add_argument("--generations", type=int,help="Number of generations the specie is evolved through", default=5000)
    parser.add_argument("--size",type=tuple, default=None, help="Resize the image with tuple of format: (y,x)")
    parser.add_argument("--save",type=str, help="Save image with a name", default=None)
    args = parser.parse_args()

    target_image = load_target_image(args.image,size=args.size)

    evolution = Evolution(target_image,genes=args.genes)
    evolution.evolve(args.generations)

    if args.save != None:
        save_image(args.save,evolution.specie.phenotype)

if __name__ == "__main__":
    main()