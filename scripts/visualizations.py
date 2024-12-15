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


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

def plot_bayesian_hierarchical(mu_prior=0, sigma_prior=1, N=10):
    # Validate input
    if not isinstance(mu_prior, (int, float)):
        raise ValueError("mu_prior must be a numeric value.")
    if not isinstance(sigma_prior, (int, float)):
        raise ValueError("sigma_prior must be a numeric value.")
    if sigma_prior <= 0:
        raise ValueError("sigma_prior must be greater than 0.")

    # Simulate data
    X = np.linspace(0, 10, N)
    true_slope = 2
    true_intercept = 1
    y_true = true_slope * X + true_intercept + np.random.normal(0, 1, N)
    
    # Define priors
    slope_prior = norm(mu_prior, sigma_prior)
    
    # Sample from the prior for slope and intercept
    slope_samples = slope_prior.rvs(1000)
    intercept_samples = np.random.normal(0, 1, 1000)  # Prior for intercept
    
    # Create images for each plot
    images = []
    
    # No Pooling visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y_true, c='blue', label='Observed Data')
    ax.plot(X, true_slope * X + true_intercept, '--r', label='True Model')
    ax.set_title("No Pooling")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    # Save No Pooling plot to images list
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    images.append(img.read())  # Add the image to the list

    # Partial Pooling visualization with Gradient Effect
    fig, ax = plt.subplots(figsize=(10, 6))
    partial_slope_samples = slope_samples[:N] * np.random.normal(0, 1, N)  # Simulated partial pooling effect (matching the shape)
    
    # Create a color gradient based on the variance (or another measure)
    variance = np.abs(partial_slope_samples)  # Use absolute slope for gradient demonstration
    points = np.array([X, partial_slope_samples * X + true_intercept]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm_color = plt.Normalize(variance.min(), variance.max())  # Rename the normalization variable
    lc = LineCollection(segments, cmap='viridis', norm=norm_color, linewidth=2, alpha=0.7)

    ax.add_collection(lc)
    ax.scatter(X, y_true, c='blue', label='Observed Data')
    ax.plot(X, partial_slope_samples * X + true_intercept, '--g', label='Partial Pooling Model')
    ax.set_title("Partial Pooling with Gradient")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    # Save Partial Pooling plot to images list
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    images.append(img.read())  # Add the image to the list
    
    # Complete Pooling visualization with Gradient Effect
    fig, ax = plt.subplots(figsize=(10, 6))
    complete_slope_samples = np.mean(slope_samples)  # Complete pooling (sharing the slope across all groups)
    
    # Create a gradient effect (color line based on variance)
    variance = np.abs(np.repeat(complete_slope_samples, N) - y_true)  # Simulate uncertainty in variance
    points = np.array([X, complete_slope_samples * X + true_intercept]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm_color = plt.Normalize(variance.min(), variance.max())  # Rename the normalization variable
    lc = LineCollection(segments, cmap='viridis', norm=norm_color, linewidth=2, alpha=0.7)

    ax.add_collection(lc)
    ax.scatter(X, y_true, c='blue', label='Observed Data')
    ax.plot(X, complete_slope_samples * X + true_intercept, '--m', label='Complete Pooling Model')
    ax.set_title("Complete Pooling with Gradient")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    # Save Complete Pooling plot to images list
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    images.append(img.read())  # Add the image to the list
    
    return images  # List containing three separate images
