import os
import pandas as pd
import torch
from torch import nn
from flask import Flask, render_template, request, url_for
import glob
from train_model import get_model

app = Flask(__name__)
model = get_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        wage = float(request.form['wage'])
        year = int(request.form['year'])
        lon = float(request.form['longitude'])
        lat = float(request.form['latitude'])
        
        input_data = torch.tensor([[wage, year, lon, lat]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_data)
            result = 'Certified' if prediction.item() >= 0.5 else 'Denied'
    return render_template('index.html', result=result)

@app.route('/visualizations')
def visualizations():
    images = glob.glob('static/visualizations/*.png')
    images = [os.path.basename(img) for img in images]  
    return render_template('visualizations.html', images=images)
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)