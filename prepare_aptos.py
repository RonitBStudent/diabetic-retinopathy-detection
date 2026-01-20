import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_aptos_data(csv_path, output_dir, test_size=0.2, random_state=42):
    """Prepare APTOS dataset splits."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the training CSV
    df = pd.read_csv(csv_path)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['diagnosis']  # Stratified split for class balance
    )
    
    # Save the splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print("\nClass distribution in training set:")
    print(train_df['diagnosis'].value_counts().sort_index())
    print("\nClass distribution in validation set:")
    print(val_df['diagnosis'].value_counts().sort_index())

if __name__ == "__main__":
    # Paths
    data_dir = "/Users/Ronit/Downloads/aptos2019-blindness-detection"
    output_dir = "data/aptos"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the data
    prepare_aptos_data(
        csv_path=os.path.join(data_dir, "train.csv"),
        output_dir=output_dir,
        test_size=0.2,
        random_state=42
    )
