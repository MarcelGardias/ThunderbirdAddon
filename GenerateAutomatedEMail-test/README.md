# GenerateAutomatedEMail
### Marcel Gardias, Patrick Unverricht, Janina Patzer, Daniel Fauland

### Get the data
    - Donwload data here: https://drive.google.com/drive/folders/1OQe9VRXQxIJAQGgw5Zq_diYZXpBPHOu0?usp=sharing]
    - Extract the downloaded data to "data" folder
    - Extract the amazon.zip file within the data folder

### Preprocess data
    - To preprocess the emails run 'preprocess_emails.py'
    - To preprocess the amazon data run 'preprocess_amazon.py'

### Training
    - To train the data run 'train_model.py'
    - You can specify the desired file for training ('emails.csv' or 'amazon.csv') as well as some training related parameters in the settings dictionary

### Predicting the data
    - To test the model run 'predict_model.py'
    - NOTE: The settings must be the same for training and predicting

