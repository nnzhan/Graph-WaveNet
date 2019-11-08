import argparse
import pickle
import numpy as np
from util import *
import matplotlib.pyplot as plt

def print_stats(args):
  if '/' not in args.adj_mx:
    full_path = 'data/sensor_graph/' + args.adj_mx
  else:
    full_path = args.adj_mx
  
  A = load_pickle(full_path)
  adj_mx = A[2]

  degrees = np.sum(adj_mx > 0, 1)
  print(f"Minimum Degree: {np.min(degrees)}")
  print(f"Maximum Degree: {np.max(degrees)}")
  print(f"Average Degree: {np.mean(degrees)}")

  plt.imshow(adj_mx, cmap='hot', interpolation='nearest')
  plt.title('Adjacency Matrix Heatmap')
  plt.show()

  plt.hist(degrees) 
  plt.xticks(range(0,max(degrees) + 5,5))
  plt.title('LA Degree Distribution')
  plt.xlabel('Node Degree')
  plt.ylabel('Number of Nodes')
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--adj_mx', type=str, default='adj_mat.pkl',
    help='Pickle file in data/sensor_graph containing adjacency matrix')
  args = parser.parse_args()
  print_stats(args)
