# H1B Visa Prediction & Data Storytelling

This project is a Flask-based web application that analyzes H1B visa application data. It provides interactive predictions for case status (Certified vs. Denied) using a machine learning model and presents various data visualizations to uncover trends in the H1B visa landscape.

## Features

*   **Prediction Tool**: Predict the likelihood of an H1B visa application being "Certified" or "Denied" based on:
    *   Prevailing Wage
    *   Year
    *   Location (Longitude & Latitude)
*   **Visualizations**: Explore generated plots showing trends such as:
    *   Wage distribution
    *   Top occupations and job titles
    *   Geographic distribution of applications
    *   Case status breakdown
*   **Insights**: Data-driven insights into the factors affecting H1B processing.
*   **About**: Information about the project and dataset.

## Technology Stack

*   **Python**: Core programming language.
*   **Flask**: Web framework for the application interface.
*   **PyTorch**: Deep learning framework used to train and run the neural network classifier for predictions.
*   **Pandas**: Data manipulation and analysis.
*   **Matplotlib/Seaborn** (implied): Used for generating static visualizations.

## Project Structure

*   `app.py`: The main Flask application entry point. Handles routing and model inference.
*   `train_model.py`: Script used to train the PyTorch neural network (`SimpleClassifier`) on the H1B dataset.
*   `h1b_model.pt`: Saved PyTorch model state dictionary used for making predictions.
*   `templates/`: HTML templates for the web pages (`index.html`, `visualizations.html`, `about.html`, `insights.html`).
*   `static/visualizations/`: Directory containing the generated data visualization images.
*   `cleaned_h1b_data_subset.csv`: The dataset used for training the model.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. You can verify the required packages by checking `app.py` and `train_model.py`. You effectively need:
    ```bash
    pip install flask pandas torch numpy
    ```

## Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to:
    `http://127.0.0.1:5000/`

3.  **Make a Prediction:**
    *   Enter the required details (Wage, Year, Longitude, Latitude) on the home page.
    *   Click the submit button to see if the case is predicted to be Certified or Denied.

## Model Details

The model is a simple feed-forward neural network (`SimpleClassifier`) implemented in PyTorch.
*   **Input**: 4 features (Wage, Year, Longitude, Latitude)
*   **Architecture**:
    *   Linear (Input -> 100) -> ReLU
    *   Linear (100 -> 64) -> ReLU
    *   Linear (64 -> 1) -> Sigmoid
*   **Output**: Probability of certification (Threshold: 0.5)
