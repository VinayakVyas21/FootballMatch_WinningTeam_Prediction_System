import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
def load_data(file_path, sample_size=None):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} matches")
    
    # Option to use a smaller sample for faster processing
    if sample_size and sample_size < len(df):
        # Use more recent matches for better relevance
        df = df.sort_values('date').tail(sample_size)
        print(f"Using {sample_size} most recent matches for faster processing")
    
    return df

# Preprocess the data
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Select only relevant columns
    df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 
             'tournament', 'country', 'neutral']]
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create result column
    df['result'] = np.where(df['home_score'] > df['away_score'], 'Home Win',
                   np.where(df['home_score'] == df['away_score'], 'Draw', 'Away Win'))
    
    # Convert neutral to boolean
    df['neutral'] = df['neutral'].astype(bool)
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df

# Feature engineering - simplified for faster processing
def engineer_features(df):
    print("Engineering features...")
    
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Sort by date
    data = data.sort_values('date')
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['home_team', 'away_team', 'tournament', 'country']:
        le = LabelEncoder()
        data[f'{col}_encoded'] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Calculate team statistics (simplified approach)
    print("Calculating team statistics...")
    
    # Create dictionaries to store team stats
    team_win_rates = {}
    team_goal_scored_avg = {}
    team_goal_conceded_avg = {}
    
    # Calculate overall stats for each team
    unique_teams = set(data['home_team'].unique()) | set(data['away_team'].unique())
    
    for team in unique_teams:
        # Get all matches for this team
        home_matches = data[data['home_team'] == team]
        away_matches = data[data['away_team'] == team]
        
        # Calculate win rate
        home_wins = sum(home_matches['result'] == 'Home Win')
        away_wins = sum(away_matches['result'] == 'Away Win')
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches > 0:
            win_rate = (home_wins + away_wins) / total_matches
        else:
            win_rate = 0.5  # Default
        
        # Calculate goal averages
        if len(home_matches) > 0:
            home_goals_scored_avg = home_matches['home_score'].mean()
            home_goals_conceded_avg = home_matches['away_score'].mean()
        else:
            home_goals_scored_avg = 1.0
            home_goals_conceded_avg = 1.0
        
        if len(away_matches) > 0:
            away_goals_scored_avg = away_matches['away_score'].mean()
            away_goals_conceded_avg = away_matches['home_score'].mean()
        else:
            away_goals_scored_avg = 1.0
            away_goals_conceded_avg = 1.0
        
        # Store stats in dictionaries
        team_win_rates[team] = win_rate
        team_goal_scored_avg[team] = (home_goals_scored_avg + away_goals_scored_avg) / 2
        team_goal_conceded_avg[team] = (home_goals_conceded_avg + away_goals_conceded_avg) / 2
    
    # Calculate head-to-head stats for each pair of teams
    h2h_win_rates = {}
    
    # Add team stats to the dataframe
    data['home_win_rate'] = data['home_team'].map(team_win_rates).fillna(0.5)
    data['away_win_rate'] = data['away_team'].map(team_win_rates).fillna(0.5)
    data['home_avg_goals_scored'] = data['home_team'].map(team_goal_scored_avg).fillna(1.0)
    data['home_avg_goals_conceded'] = data['home_team'].map(team_goal_conceded_avg).fillna(1.0)
    data['away_avg_goals_scored'] = data['away_team'].map(team_goal_scored_avg).fillna(1.0)
    data['away_avg_goals_conceded'] = data['away_team'].map(team_goal_conceded_avg).fillna(1.0)
    
    # Simplified h2h calculation - use a default value of 0.5 for all matches
    data['h2h_win_rate'] = 0.5
    
    print("Feature engineering complete")
    return data, label_encoders

# Prepare data for modeling
def prepare_for_modeling(data):
    print("Preparing data for modeling...")
    
    # Define features and target
    features = [
        'home_team_encoded', 'away_team_encoded', 'tournament_encoded', 'country_encoded',
        'neutral', 'home_win_rate', 'away_win_rate', 'home_avg_goals_scored',
        'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
        'h2h_win_rate'
    ]
    
    X = data[features]
    
    # Encode the target variable
    le_result = LabelEncoder()
    y = le_result.fit_transform(data['result'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, le_result

# Train the model
def train_model(X_train, y_train):
    print("Training XGBoost model...")
    
    # Define the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training complete")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test, le_result):
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    class_names = le_result.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=12)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    
    return accuracy

# Function to make predictions for new matches - simplified version
def predict_match(model, home_team, away_team, tournament, neutral, label_encoders, le_result, data):
    # Create a dataframe for the new match
    try:
        # Try to use the first country in the dataset as a default instead of 'Unknown'
        default_country = data['country'].iloc[0]
        country_encoded = label_encoders['country'].transform([default_country])[0]
    except:
        # If that fails, use the most common country
        default_country = data['country'].mode()[0]
        country_encoded = label_encoders['country'].transform([default_country])[0]
    
    new_match = pd.DataFrame({
        'home_team_encoded': [label_encoders['home_team'].transform([home_team])[0]],
        'away_team_encoded': [label_encoders['away_team'].transform([away_team])[0]],
        'tournament_encoded': [label_encoders['tournament'].transform([tournament])[0]],
        'country_encoded': [country_encoded],
        'neutral': [neutral],
    })
    
    # Get team stats from the data
    # For home team
    home_team_matches = data[data['home_team'] == home_team]
    away_team_matches = data[data['away_team'] == away_team]
    
    # Calculate win rates
    if len(home_team_matches) > 0 or len(data[data['away_team'] == home_team]) > 0:
        home_wins = sum(home_team_matches['result'] == 'Home Win')
        away_wins = sum(data[(data['away_team'] == home_team) & (data['result'] == 'Away Win')]['result'] == 'Away Win')
        total_matches = len(home_team_matches) + len(data[data['away_team'] == home_team])
        home_win_rate = (home_wins + away_wins) / total_matches if total_matches > 0 else 0.5
    else:
        home_win_rate = 0.5
    
    if len(away_team_matches) > 0 or len(data[data['home_team'] == away_team]) > 0:
        away_wins = sum(away_team_matches['result'] == 'Away Win')
        home_wins = sum(data[(data['home_team'] == away_team) & (data['result'] == 'Home Win')]['result'] == 'Home Win')
        total_matches = len(away_team_matches) + len(data[data['home_team'] == away_team])
        away_win_rate = (away_wins + home_wins) / total_matches if total_matches > 0 else 0.5
    else:
        away_win_rate = 0.5
    
    # Calculate goal averages
    if len(home_team_matches) > 0:
        home_avg_goals_scored = home_team_matches['home_score'].mean()
        home_avg_goals_conceded = home_team_matches['away_score'].mean()
    else:
        home_avg_goals_scored = 1.0
        home_avg_goals_conceded = 1.0
    
    if len(away_team_matches) > 0:
        away_avg_goals_scored = away_team_matches['away_score'].mean()
        away_avg_goals_conceded = away_team_matches['home_score'].mean()
    else:
        away_avg_goals_scored = 1.0
        away_avg_goals_conceded = 1.0
    
    # Add stats to the new match
    new_match['home_win_rate'] = home_win_rate
    new_match['away_win_rate'] = away_win_rate
    new_match['home_avg_goals_scored'] = home_avg_goals_scored
    new_match['home_avg_goals_conceded'] = home_avg_goals_conceded
    new_match['away_avg_goals_scored'] = away_avg_goals_scored
    new_match['away_avg_goals_conceded'] = away_avg_goals_conceded
    new_match['h2h_win_rate'] = 0.5  # Simplified approach
    
    # Make prediction
    pred_proba = model.predict_proba(new_match)[0]
    pred_class = model.predict(new_match)[0]
    result = le_result.inverse_transform([pred_class])[0]
    
    # Return prediction and probabilities
    return {
        'predicted_result': result,
        'probabilities': {
            le_result.classes_[i]: f"{prob:.2f}" for i, prob in enumerate(pred_proba)
        }
    }

# Main function
def main():
    # Load and preprocess data - using a smaller sample for faster processing
    sample_size = 10000  # Use last 10,000 matches for faster processing
    df = load_data('international_results/results.csv', sample_size)
    df = preprocess_data(df)
    
    # Engineer features
    data, label_encoders = engineer_features(df)
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test, le_result = prepare_for_modeling(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, le_result)
    
    # Example prediction
    print("\nExample prediction:")
    prediction = predict_match(
        model, 'Brazil', 'Argentina', 'Friendly', False, 
        label_encoders, le_result, data
    )
    print(f"Predicted result: {prediction['predicted_result']}")
    print("Probabilities:")
    for result, prob in prediction['probabilities'].items():
        print(f"  {result}: {prob}")
    
    return model, label_encoders, le_result, data

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()