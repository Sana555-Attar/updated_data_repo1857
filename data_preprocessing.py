import os.path
from pathlib import Path
import json
import pandas as pd
from clearml import Dataset, Task

task = Task.init(
    project_name='sarcasm_detector',
    task_name='preprocess data',
    task_type='data_processing',
    reuse_last_task_id=False
)
#DATA SOURCE 
data_source = 'Sarcasm_Headlines_Dataset.json'
sentences = []
labels = []


def get_csv(file_path):
    for item in open(file_path, 'r'):
        item = json.loads(item)
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    data = pd.DataFrame({'headline':sentences, 'is_sarcastic':labels })
    train = data.iloc[:int(len(data)*0.9),:]
    test = data.iloc[int(len(data)*0.90):,:]
    
    return train, test


train, test = get_csv('Sarcasm_Headlines_Dataset.json')
print(train.shape)
train.to_csv('data/train.csv',index=False)
test.to_csv('data/test.csv',index=False)


# Create the folder we'll output the preprocessed data into
preprocessed_data_folder = Path('data')
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)

# Get the dataset
dataset = Dataset.create(
    dataset_name='sarcasm_dataset',
    dataset_project='sarcasm_detector'
    
)

dataset.add_files(preprocessed_data_folder / 'train.csv')
dataset.add_files(preprocessed_data_folder / 'test.csv')
dataset.get_logger().report_table(title='Train data', series='head', table_plot=train.head())
dataset.get_logger().report_table(title='test data', series='head', table_plot=test.head())
dataset.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {dataset.id}")
