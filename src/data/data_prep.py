import pandas as pd
import os


def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")
#train_data = pd.read_csv(os.path.join("data", "raw", "train.csv"))
#test_data = pd.read_csv(os.path.join("data", "raw", "test.csv"))

def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.columns:
            median = df[col].median()
            df.fillna({col:median}, inplace=True)
        return df
    except Exception as e:
        raise Exception("Errror filling missing values: {e}")

def save_data(df : pd.DataFrame, filepath : str) -> None:
    try:
       df.to_csv(filepath, index = False)
    except Exception as e:
        raise Exception(f"Error in saving file to {filepath}")
    
    
# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

def main():
    try:
        raw_data_path = os.path.join("data", "raw")
        processed_data_path = os.path.join("data", "processed")

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        train_processed_data = fill_missing_with_median(train_data)
        test_processed_data = fill_missing_with_median(test_data)

    #data_path = os.path.join("data", "processed")
        os.makedirs(processed_data_path)

        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed.csv"))          
    except Exception as e:
        raise Exception(f"An error occured {e}")
    
if __name__ =="__main__":
    main()


