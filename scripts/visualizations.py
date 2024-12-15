import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plot_hierarchical_model(group_mean, group_variance, individual_variance):
    # Define the hierarchical model (simplified)
    x = np.linspace(-10, 10, 100)
    group_distribution = (1 / np.sqrt(2 * np.pi * group_variance)) * np.exp(-(x - group_mean)**2 / (2 * group_variance))
    individual_distribution = (1 / np.sqrt(2 * np.pi * individual_variance)) * np.exp(-(x - group_mean)**2 / (2 * individual_variance))
    
    # Plot the group and individual distributions
    plt.figure(figsize=(8, 6))
    plt.plot(x, group_distribution, label="Group Distribution", color='blue')
    plt.plot(x, individual_distribution, label="Individual Distribution", color='red')
    plt.title("Hierarchical Model")
    plt.legend()
    plt.xlabel("Parameter Value")
    plt.ylabel("Density")
    
    # Convert the plot to a PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img.read()


import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plot_bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance):
    # Define Bayesian update logic for the prior and likelihood
    x = np.linspace(-10, 10, 100)
    prior = (1 / np.sqrt(2 * np.pi * prior_variance)) * np.exp(-(x - prior_mean)**2 / (2 * prior_variance))
    likelihood = (1 / np.sqrt(2 * np.pi * likelihood_variance)) * np.exp(-(x - likelihood_mean)**2 / (2 * likelihood_variance))
    posterior = prior * likelihood  # Unnormalized posterior
    
    # Normalize the posterior
    posterior /= np.sum(posterior)
    
    # Plot the prior, likelihood, and posterior
    plt.figure(figsize=(8, 6))
    plt.plot(x, prior, label="Prior", color='blue')
    plt.plot(x, likelihood, label="Likelihood", color='red')
    plt.plot(x, posterior, label="Posterior", color='green')
    plt.title("Bayesian Update")
    plt.legend()
    plt.xlabel("Parameter Value")
    plt.ylabel("Density")
    
    # Convert the plot to a PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img.read()
