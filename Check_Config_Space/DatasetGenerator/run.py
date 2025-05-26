from FileIO import *
from Generator import *

import argparse
import os

parser = argparse.ArgumentParser(description="Generate dataset for config space")
parser.add_argument('-s', '--size', type=float, nargs=2, default=[1, 1], help='Boundary size of the scene')
parser.add_argument('-p', '--points', type=int, default=20, help='Number of view points')
parser.add_argument('-r', '--rate', type=float, default=0.1, help='Sample rate of the config space')
parser.add_argument('-n', '--num', type=int, default=16, help='Number of scenes to generate')
parser.add_argument('-d', '--dir', type=str, default='../data', help='Directory to save the dataset')
parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use for generation')

args = parser.parse_args()

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    dir = os.path.join(os.path.dirname(__file__), args.dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Initialize the generator
    generator = Generator(
        boundary_size=args.size,
        view_point_num=args.points,
        sample_rate=args.rate,
        generate_scene_num=args.num
    )
    # Generate the dataset
    dataset = generator.generate(thread_num=args.threads)
    # Save the dataset
    save_pkl(dataset, dir)