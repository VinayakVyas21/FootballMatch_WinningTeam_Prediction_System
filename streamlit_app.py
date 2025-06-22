import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from football_predictor import predict_match, load_data, preprocess_data, engineer_features, prepare_for_modeling, train_model

# Set page title and layout
st.set_page_config(page_title="Football Match Predictor", layout="wide")

# Title and description
st.title("International Football Match Outcome Predictor")
st.markdown("""
This app predicts the outcome of international football matches using historical data and XGBoost machine learning model.

The model is trained on international football match results from 1872 to present, considering factors like:
- Team performance history
- Head-to-head records
- Tournament context
- Home advantage
""")

# Function to load or train the model
@st.cache_resource
def get_model_and_encoders():
    model_file = 'football_model.pkl'
    encoders_file = 'label_encoders.pkl'
    result_encoder_file = 'result_encoder.pkl'
    data_file = 'processed_data.pkl'
    
    # Check if model files exist
    if os.path.exists(model_file) and os.path.exists(encoders_file) and os.path.exists(result_encoder_file) and os.path.exists(data_file):
        # Load existing model and encoders
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_file, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(result_encoder_file, 'rb') as f:
            le_result = pickle.load(f)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        # Train new model
        with st.spinner('Training model for the first time. This may take several minutes...'):
            # Load and preprocess data
            df = load_data('international_results/results.csv')
            df = preprocess_data(df)
            
            # Engineer features
            data, label_encoders = engineer_features(df)
            
            # Prepare for modeling
            X_train, X_test, y_train, y_test, le_result = prepare_for_modeling(data)
            
            # Train the model
            model = train_model(X_train, y_train)
            
            # Save model and encoders
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            with open(encoders_file, 'wb') as f:
                pickle.dump(label_encoders, f)
            with open(result_encoder_file, 'wb') as f:
                pickle.dump(le_result, f)
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    
    return model, label_encoders, le_result, data

# Get unique teams and tournaments
def get_unique_values(data):
    teams = sorted(list(set(data['home_team'].unique()) | set(data['away_team'].unique())))
    tournaments = sorted(data['tournament'].unique())
    return teams, tournaments

# Main app function
def main():
    # Load or train model
    model, label_encoders, le_result, data = get_model_and_encoders()
    
    # Get unique teams and tournaments
    teams, tournaments = get_unique_values(data)
    
    # Create sidebar for inputs
    st.sidebar.header("Match Details")
    
    # Team selection
    home_team = st.sidebar.selectbox("Home Team", teams)
    away_team = st.sidebar.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
    
    # Tournament selection
    tournament = st.sidebar.selectbox("Tournament", tournaments)
    
    # Neutral venue
    neutral = st.sidebar.checkbox("Neutral Venue")
    
    # Predict button
    if st.sidebar.button("Predict Match Outcome"):
        # Check if teams are the same
        if home_team == away_team:
            st.error("Please select different teams for home and away.")
        else:
            # Make prediction
            with st.spinner('Predicting match outcome...'):
                prediction = predict_match(
                    model, home_team, away_team, tournament, neutral, 
                    label_encoders, le_result, data
                )
            
            # Display prediction
            st.subheader(f"Match: {home_team} vs {away_team}")
            st.write(f"Tournament: {tournament}")
            st.write(f"Venue: {'Neutral' if neutral else 'Home'}")
            
            # Create columns for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predicted Outcome")
                st.markdown(f"<h1 style='text-align: center;'>{prediction['predicted_result']}</h1>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Outcome Probabilities")
                # Convert probabilities to float for visualization
                probs = {k: float(v) for k, v in prediction['probabilities'].items()}
                
                # Create a DataFrame for the chart
                prob_df = pd.DataFrame({
                    'Outcome': list(probs.keys()),
                    'Probability': list(probs.values())
                })
                
                # Sort by probability
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                # Display as bar chart
                st.bar_chart(prob_df.set_index('Outcome'))
            
            # Display historical context
            st.subheader("Historical Context")
            
            # Get head-to-head matches
            h2h_matches = data[
                ((data['home_team'] == home_team) & (data['away_team'] == away_team)) | 
                ((data['home_team'] == away_team) & (data['away_team'] == home_team))
            ].sort_values('date', ascending=False)
            
            if len(h2h_matches) > 0:
                st.write(f"These teams have played {len(h2h_matches)} times before.")
                
                # Calculate stats
                home_team_wins = sum(
                    ((h2h_matches['home_team'] == home_team) & (h2h_matches['result'] == 'Home Win')) | 
                    ((h2h_matches['away_team'] == home_team) & (h2h_matches['result'] == 'Away Win'))
                )
                away_team_wins = sum(
                    ((h2h_matches['home_team'] == away_team) & (h2h_matches['result'] == 'Home Win')) | 
                    ((h2h_matches['away_team'] == away_team) & (h2h_matches['result'] == 'Away Win'))
                )
                draws = len(h2h_matches) - home_team_wins - away_team_wins
                
                st.write(f"{home_team} wins: {home_team_wins}")
                st.write(f"{away_team} wins: {away_team_wins}")
                st.write(f"Draws: {draws}")
                
                # Show recent matches
                st.write("Recent matches:")
                recent_matches = h2h_matches.head(5)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']]
                recent_matches['date'] = recent_matches['date'].dt.date
                st.dataframe(recent_matches)
            else:
                st.write("These teams have never played each other before.")
    
    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Model")
    st.sidebar.info("""
    This prediction model uses XGBoost, a powerful machine learning algorithm.
    
    Features used include:
    - Team encoding
    - Tournament type
    - Home advantage
    - Team's recent performance
    - Head-to-head history
    - Goal scoring patterns
    
    The model was trained on international football matches from 1872 to present.
    """)

# Run the app
if __name__ == "__main__":
    main()