import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_generation.generate_ticket_data import TicketDataGenerator
from models.fcr_predictor import FCRPredictor

def create_output_directories():
    """Create necessary directories for outputs"""
    directories = ['data', 'models', 'results', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_data():
    """Generate synthetic support ticket data"""
    generator = TicketDataGenerator()
    df = generator.save_data('data/support_tickets.csv')
    return df

def analyze_data(df):
    """Perform exploratory data analysis"""
    # FCR distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='fcr')
    plt.title('Distribution of First Contact Resolution')
    plt.savefig('visualizations/fcr_distribution.png')
    plt.close()
    
    # FCR by category
    plt.figure(figsize=(12, 6))
    fcr_by_category = df.groupby('category')['fcr'].mean().sort_values(ascending=False)
    fcr_by_category.plot(kind='bar')
    plt.title('FCR Rate by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/fcr_by_category.png')
    plt.close()
    
    # FCR by channel
    plt.figure(figsize=(8, 6))
    fcr_by_channel = df.groupby('channel')['fcr'].mean().sort_values(ascending=False)
    fcr_by_channel.plot(kind='bar')
    plt.title('FCR Rate by Channel')
    plt.tight_layout()
    plt.savefig('visualizations/fcr_by_channel.png')
    plt.close()
    
    # Resolution time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='resolution_time', bins=50)
    plt.title('Distribution of Resolution Time')
    plt.xlabel('Resolution Time (hours)')
    plt.savefig('visualizations/resolution_time_distribution.png')
    plt.close()

def train_and_evaluate_model(df):
    """Train and evaluate the FCR prediction model"""
    fcr_predictor = FCRPredictor()
    results = fcr_predictor.train(df, model_type='random_forest')
    
    # Save model
    fcr_predictor.save_model('models/fcr_model.joblib')
    
    # Save feature importance plot
    feature_importance = fcr_predictor.get_feature_importance()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for FCR Prediction')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    
    return fcr_predictor, results

def main():
    """Main execution function"""
    print("Starting FCR Analysis Pipeline...")
    
    # Create directories
    create_output_directories()
    
    # Generate data
    print("\nGenerating synthetic support ticket data...")
    df = generate_data()
    
    # Analyze data
    print("\nPerforming exploratory data analysis...")
    analyze_data(df)
    
    # Train and evaluate model
    print("\nTraining FCR prediction model...")
    fcr_predictor, results = train_and_evaluate_model(df)
    
    print("\nPipeline completed successfully!")
    print("- Data saved to: data/support_tickets.csv")
    print("- Model saved to: models/fcr_model.joblib")
    print("- Visualizations saved to: visualizations/")

if __name__ == "__main__":
    main() 