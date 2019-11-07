import argparse
import pickle
import numpy as np

def print_stats(args):
  with open('data/sensor_graph/' + args.adj_mx, 'rb') as f:
    A = pickle.load(f)

  adj_mx = A[2]
  degrees = np.sum(adj_mx > 0, 1)

  print(f"Minimum Degree: {np.min(degrees)}")
  print(f"Maximum Degree: {np.max(degrees)}")
  print(f"Average Degree: {np.mean(degrees)}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--adj_mx', type=str, default='adj_mat.pkl',
    help='Pickle file in data/sensor_graph containing adjacency matrix')
  args = parser.parse_args()
  print_stats(args)
