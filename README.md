# International Football Match Predictor

A machine learning system that predicts the outcomes of international football matches using historical data from 1872 to present. The system uses XGBoost, a powerful gradient boosting algorithm, to classify match results into three categories: Home Win, Draw, or Away Win.

## Features

- **Predictive Analysis**: Forecasts match outcomes based on historical data with probability scores
- **Feature Engineering**: Utilizes team performance metrics, head-to-head records, and contextual factors
- **Interactive Interface**: Provides a user-friendly Streamlit web application for making predictions
- **Comprehensive Dataset**: Uses 47,000+ international football matches from 1872 to present
- **Visual Insights**: Displays prediction probabilities and historical context between teams
- **Performance Metrics**: Includes model evaluation with accuracy scores and confusion matrices

## Dataset

The system uses the international football results dataset from [martj42's GitHub repository](https://github.com/martj42/international_results), which includes:

- 47,000+ international football matches from 1872 to present
- Match details including teams, scores, tournament, location, etc.
- Men's full international matches only (no Olympic Games or B-teams)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Football_Pridiction_System.git
   cd Football_Pridiction_System
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```
   git clone https://github.com/martj42/international_results.git
   ```

## Usage

### Running the Prediction Model

To train the model and make predictions using the Python script:

```
python football_predictor.py
```

This will:
- Load and preprocess the data
- Engineer features
- Train an XGBoost classifier
- Evaluate the model
- Make an example prediction (Brazil vs Argentina)

### Using the Streamlit Web Interface

To launch the interactive web interface:

```
streamlit run streamlit_app.py
```

This will open a browser window where you can:
1. Select the home and away teams
2. Choose a tournament
3. Specify if the match is at a neutral venue
4. Get predictions and view historical context

## Model Details

### Feature Engineering

The system creates the following features:
- Team encoding (using label encoding)
- Tournament and country encoding
- Neutral venue indicator
- Team win rates (calculated from historical performance)
- Goal scoring and conceding averages
- Head-to-head statistics

### Model Evaluation

The model is evaluated using:
- Accuracy score
- Precision, recall, and F1-score for each class
- Confusion matrix visualization
- Feature importance analysis

## Project Structure

- `football_predictor.py`: Main Python script for data processing and model training
- `streamlit_app.py`: Streamlit web interface
- `requirements.txt`: List of required Python packages
- `international_results/`: Directory containing the dataset
- `football_model.pkl`: Saved trained model
- `label_encoders.pkl`: Saved label encoders for categorical variables
- `result_encoder.pkl`: Saved encoder for result labels
- `processed_data.pkl`: Saved processed dataset
- `confusion_matrix.png`: Visualization of model performance
- `feature_importance.png`: Visualization of feature importance
- `README.md`: Project documentation
- `explanation.txt`: Detailed explanation of code functionality

## Technical Implementation

### Data Processing Pipeline

1. **Data Loading**: Loads the CSV dataset with option to sample for faster processing
2. **Preprocessing**: Cleans data, converts date formats, and creates target variable
3. **Feature Engineering**: Encodes categorical variables and calculates team statistics
4. **Model Training**: Trains XGBoost classifier on processed data
5. **Evaluation**: Assesses model performance with various metrics

### Streamlit Application

The web interface provides:
- Team and tournament selection dropdowns
- Neutral venue toggle
- Prediction results with probability visualization
- Historical head-to-head statistics
- Recent match history between selected teams

## Future Improvements

- Incorporate more advanced features like team rankings and player statistics
- Add time-based weighting to give more importance to recent matches
- Implement model hyperparameter tuning for better performance
- Expand the interface to include more visualization options
- Add support for club football matches

## Acknowledgements

- Dataset provided by [martj42](https://github.com/martj42/international_results)
- Built using XGBoost, scikit-learn, pandas, and Streamlit