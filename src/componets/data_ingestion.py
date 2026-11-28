
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.expesion import CustomException
from src.componets.logger import logging

from src.componets.data_transformation import DataTransformation
from src.componets.data_transformation import DataTransformationConfig
from src.componets.model_trainer import ModelTrainerConnfig
from src.componets.model_trainer import ModelTrainer
from src.utils import save_object






# ==========================================
# DATA INGESTION CONFIGURATION
# ==========================================

@dataclass
class DataIngestionConfig:

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    train_data_path: str = os.path.join(
        project_root, "artifacts", "train.csv"
    )

    test_data_path: str = os.path.join(
        project_root, "artifacts", "test.csv"
    )

    raw_data_path: str = os.path.join(
        project_root, "artifacts", "raw.csv"
    )

    # Correct dataset location according to your VS Code
    data_path: str = os.path.join(
        project_root, "noteboock", "new_data", "stud.csv"
    )


# ==========================================
# DATA INGESTION COMPONENT
# ==========================================

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:

            # Step 1: Check dataset path
            logging.info("Checking dataset path")

            print("Dataset path:", self.ingestion_config.data_path)

            if not os.path.exists(self.ingestion_config.data_path):
                raise FileNotFoundError(
                    f"Dataset not found at: "
                    f"{self.ingestion_config.data_path}"
                )

            logging.info("Dataset path verified successfully")

            # Step 2: Read dataset
            df = pd.read_csv(
                self.ingestion_config.data_path
            )

            logging.info("Dataset loaded successfully")

            print("Dataset shape:", df.shape)

            # Step 3: Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            logging.info("Artifacts directory created")

            # Step 4: Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Raw dataset saved successfully")

            # Step 5: Train test split
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Step 6: Save training dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Step 7: Save testing dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            print("\nData Ingestion Completed!")
            print("Total dataset:", len(df))
            print("Training dataset:", len(train_set))
            print("Testing dataset:", len(test_set))

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logging.error("Error occurred during data ingestion")

            raise CustomException(e, sys)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":


    obj = DataIngestion()

    train_data,test_data = obj.initiate_data_ingestion()

    print("\nTrain data path:", train_data)
    print("Test data path:", test_data)

    data_transformation = DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array))
