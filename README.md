# Team SAHARA Kyn Project - Content Moderation Track

1. Download the repository or clone it using the following command: git clone
2. Install the required Python libraries listed in requirements.txt: pip install -r requirements.txt
   (Optional) Run Data_Preprocessing.ipynb to preprocess the dataset and generate the preprocessed_data.csv file.
   (Optional) Open and run LSTM.ipynb to train and evaluate the LSTM model for text classification using the preprocessed data.
   (Optional) Open and run BERT.ipynb to train and evaluate the BERT model for text classification using the preprocessed data.
3. Or down load our pre trained model from Drive link : https://drive.google.com/drive/folders/1jsmjbYKpQZ6pihJgOrXeirhpnxq8U0Of?usp=sharing
4. Copy the folder 'bert_model' to the parent folder where the 'Flask_App' and requirements.txt files are located.
5. Navigate to the Flask_app directory and run the Flask application by executing the following command: python main.py
6. Access the Flask application through a web browser by visiting http://localhost:5000/. You can input text or upload images to receive classification results based on the content provided.
