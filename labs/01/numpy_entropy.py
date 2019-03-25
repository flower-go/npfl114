#!/usr/bin/env python3
import numpy as np

def comp_entropy(v1, v2):
    e = -1 *np.sum( v1 * np.log(v2))   
    return e

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    d = []
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            d.append(line)
            # TODO: process the line, aggregating using Python data structures
    
    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    # Load model distribution, each line `word \t probability`.
    model_dist = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            name,probab = line.split("\t")
            model_dist[name] = float(probab)
            # TODO: process the line, aggregating using Python data structures

    # TODO: Create a NumPy array containing the model distribution.

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication). 
    y = np.unique(d, return_counts = True)
    y_nove = y[1]/len(d)
    probabs = y_nove
    entropy = comp_entropy(probabs,probabs)
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    y_zip = zip(y[0],y_nove)
    data_dict = dict(y_zip)
    
    new_model = dict.fromkeys(data_dict.keys(), 0.0)
    new_model.update(model_dist)
    
    d = dict.fromkeys(new_model.keys(), 0.0)
    d.update(data_dict)
    data_array = list(d.values())
    model_array = list(new_model.values())
      
    if any (x == 0.0 for x in model_array):
        cross_entropy = np.inf
        kl_divergence = np.inf
    else:
        cross_entropy = comp_entropy(data_array,model_array)
        kl_divergence= cross_entropy - entropy
    print("{:.2f}".format(cross_entropy))
    # and KL-divergence D_KL(data distribution, model_distribution)
   
    print("{:.2f}".format(kl_divergence))