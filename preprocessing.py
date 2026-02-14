import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MobileDataEngine:
    """Advanced data validation engine focusing on Nulls, Duplicates, and Range checks."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_and_validate(self) -> bool:
        """Loads dataset and executes a 3-tier validation suite."""
        try:
            self.df = pd.read_csv(self.input_path)
            logging.info(f"Dataset Loaded: {self.df.shape[0]} rows.")

            # --- 1. NULL VALUE CHECK ---
            null_count = self.df.isnull().sum().sum()
            if null_count > 0:
                logging.warning(f"Null Check: {null_count} missing values found. Imputing with median.")
                self.df.fillna(self.df.median(), inplace=True)
            else:
                logging.info("Null Check: Passed (No missing values).")

            # --- 2. DUPLICATE CHECK ---
            duplicate_count = self.df.duplicated().sum()
            if duplicate_count > 0:
                logging.warning(f"Duplicate Check: Found {duplicate_count} rows. Removing duplicates.")
                self.df.drop_duplicates(inplace=True)
            else:
                logging.info("Duplicate Check: Passed (No duplicates found).")

            # --- 3. RANGE & LOGIC VALIDATION ---
            # We check if physical dimensions or battery power fall into impossible ranges
            # Example: px_height and sc_w should not be 0 or negative.
            range_issues = (self.df['px_height'] <= 0).sum() + (self.df['sc_w'] <= 0).sum()
            if range_issues > 0:
                logging.warning(f"Range Check: {range_issues} rows contain invalid zero/negative dimensions. Correcting...")
                self.df['px_height'] = self.df['px_height'].replace(0, self.df['px_height'].median())
                self.df['sc_w'] = self.df['sc_w'].replace(0, self.df['sc_w'].median())
            
            # Ensuring RAM is within a logical positive range
            if (self.df['ram'] <= 0).any():
                 self.df['ram'] = self.df['ram'].clip(lower=self.df['ram'].median())

            return True
        except Exception as e:
            logging.error(f"Validation Suite Failed: {e}")
            return False

    def generate_assignment_plots(self):
        """Generates the requested Class Balance and Feature Importance visuals."""
        sns.set_theme(style="white")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Class Balance
        sns.countplot(x='price_range', data=self.df, palette='viridis', hue='price_range', legend=False, ax=axes[0])
        axes[0].set_title('Class Balance (Price Range)')

        # Feature Correlation with Target
        correlations = self.df.corr()['price_range'].sort_values(ascending=False).drop('price_range')
        correlations.plot(kind='bar', color='skyblue', ax=axes[1])
        axes[1].set_title('Feature Importance (Correlation with Price)')

        plt.tight_layout()
        plt.show()

    def export_cleaned_data(self, output_name: str = 'train_processed.csv'):
        """Exports validated data to a new file to maintain lineage."""
        self.df.to_csv(output_name, index=False)
        logging.info(f"Validated dataset exported as: {output_name}")

if __name__ == "__main__":
    engine = MobileDataEngine('train.csv')
    if engine.load_and_validate():
        engine.generate_assignment_plots()
        engine.export_cleaned_data()
#make sure validation is handled and data is prepared without nulls