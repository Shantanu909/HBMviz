from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from HBMviz.scripts.visualizations import plot_bayesian_update, plot_hierarchical_model
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



from flask import render_template, jsonify, request
import base64
from scripts.visualizations import plot_bayesian_hierarchical

@app.route('/bayesian_hierarchical', methods=['GET', 'POST'])
def pooling_methods():
    if request.method == 'POST':
        data = request.get_json()
        mu = float(data['mu'])
        sigma = float(data['sigma'])
        
        # Generate the pooling methods plot
        images = plot_bayesian_hierarchical(mu_prior=mu, sigma_prior=sigma)
        
        # Convert the plot images to base64 and return them
        img_str_no_pooling = base64.b64encode(images[0]).decode('utf-8')
        img_str_partial_pooling = base64.b64encode(images[1]).decode('utf-8')
        img_str_complete_pooling = base64.b64encode(images[2]).decode('utf-8')
        
        return jsonify({
            'img_str_no_pooling': img_str_no_pooling,
            'img_str_partial_pooling': img_str_partial_pooling,
            'img_str_complete_pooling': img_str_complete_pooling
        })
    
    # Default values for the GET request
    mu = 0
    sigma = 1

    # Generate and return the plot
    images = plot_bayesian_hierarchical(mu_prior=mu, sigma_prior=sigma)

    # Convert images to base64 for embedding
    img_str_no_pooling = base64.b64encode(images[0]).decode('utf-8')
    img_str_partial_pooling = base64.b64encode(images[1]).decode('utf-8')
    img_str_complete_pooling = base64.b64encode(images[2]).decode('utf-8')

    return render_template(
        'bayesian_hierarchical.html',
        img_str_no_pooling=img_str_no_pooling,
        img_str_partial_pooling=img_str_partial_pooling,
        img_str_complete_pooling=img_str_complete_pooling
    )


if __name__ == '__main__':
    app.run(debug=True)
