from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scripts.visualizations import plot_bayesian_update, plot_hierarchical_model
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bayesian_inference', methods=['GET', 'POST'])
def bayesian_inference():
    if request.method == 'POST':
        # Get data from sliders in the POST request
        data = request.get_json()
        prior_mean = float(data['prior_mean'])
        prior_variance = float(data['prior_variance'])
        likelihood_mean = float(data['likelihood_mean'])
        likelihood_variance = float(data['likelihood_variance'])
        
        # Generate the Bayesian update plot with the updated parameters
        img = plot_bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance)
        
        # Convert the plot to base64
        img_str = base64.b64encode(img).decode('utf-8')
        
        return jsonify({'img_str': img_str})
    
    # Default values for the GET request
    prior_mean = 0
    prior_variance = 1
    likelihood_mean = 0
    likelihood_variance = 1

    # Generate and return the plot
    img = plot_bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance)
    img_str = base64.b64encode(img).decode('utf-8')
    
    return render_template('bayesian_inference.html', img_str=img_str)

@app.route('/hierarchical_models', methods=['GET', 'POST'])
def hierarchical_models():
    if request.method == 'POST':
        # Get slider data from the POST request
        data = request.get_json()
        group_mean = float(data['group_mean'])
        group_variance = float(data['group_variance'])
        individual_variance = float(data['individual_variance'])
        
        # Generate the hierarchical model plot with the updated parameters
        img = plot_hierarchical_model(group_mean, group_variance, individual_variance)
        
        # Convert the plot to base64
        img_str = base64.b64encode(img).decode('utf-8')
        
        return jsonify({'img_str': img_str})
    
    # Default values for the GET request
    group_mean = 0
    group_variance = 1
    individual_variance = 1

    # Generate and return the plot
    img = plot_hierarchical_model(group_mean, group_variance, individual_variance)
    img_str = base64.b64encode(img).decode('utf-8')
    
    return render_template('hierarchical_models.html', img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
