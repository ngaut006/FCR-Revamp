import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import shap
import joblib

class FCRPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_columns = [
            'category', 'subcategory', 'priority', 'channel',
            'agent_team', 'agent_experience', 'resolution_time'
        ]
        
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        X = df.copy()
        
        # Convert datetime to features
        X['hour_of_day'] = pd.to_datetime(X['created_at']).dt.hour
        X['day_of_week'] = pd.to_datetime(X['created_at']).dt.dayofweek
        
        # Add these new features to feature columns
        self.feature_columns.extend(['hour_of_day', 'day_of_week'])
        
        # Encode categorical variables
        categorical_columns = ['category', 'subcategory', 'priority', 'channel', 'agent_team']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column])
            else:
                X[column] = self.label_encoders[column].transform(X[column])
        
        return X[self.feature_columns]
    
    def train(self, df, model_type='random_forest'):
        """Train the FCR prediction model"""
        X = self.preprocess_data(df)
        y = df['fcr']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Generate SHAP values for feature importance
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        
        # Store feature importance
        if model_type == 'random_forest':
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'shap_values': shap_values
        }
    
    def predict(self, ticket_data):
        """Predict FCR probability for new tickets"""
        X = self.preprocess_data(ticket_data)
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        return pd.DataFrame({
            'ticket_id': ticket_data['ticket_id'],
            'fcr_probability': probabilities[:, 1],
            'predicted_fcr': predictions
        })
    
    def get_feature_importance(self):
        """Return feature importance analysis"""
        return self.feature_importance
    
    def save_model(self, filepath='fcr_model.joblib'):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fcr_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print("Model loaded successfully")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('support_tickets.csv')
    
    # Initialize and train model
    fcr_predictor = FCRPredictor()
    results = fcr_predictor.train(df, model_type='random_forest')
    
    # Save model
    fcr_predictor.save_model()
    
    # Print feature importance
    print("\nFeature Importance:")
    print(fcr_predictor.get_feature_importance()) 