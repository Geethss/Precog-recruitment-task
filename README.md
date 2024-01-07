# Precog-recruitment-task
This github consists of two GoogleColab notebooks and a dataset. The notebooks are implemented in Python and utilize various libraries for completing the given natural language processing tasks.
## Directory Structure
Code for all the tasks are split into two notebooks.<br>

The first notebook contains the code for the word and phrase similarity tasks.
<br>Link: https://colab.research.google.com/drive/1aOP2eqft6n7k0c7Dn9ErX77VoMGEE8dA?usp=sharing <br>

The second notebook contains the code for the sentence similarity task and Bonus task. 
<br>Link: https://colab.research.google.com/drive/1lUTaNon5QtpggYEFYyGlrH4nebD5X5X-?usp=sharing 

## Commands to run the project

Instructions for the first notebook-
<br>For the word similarity task, You need to upload the 'Simlex-999' dataset provided in the files section.<br>
The first block of code can be used to cross-check if the file has been uploaded properly.
<br>Run the second block to load the Word2Vec 'GoogleNews-vectors-negative300' model.
<br>After completing the above steps, you can run the codes related to the task.<br>
In the next section titled Phrase Similarity, there are no pre-requisite code blocks to run.
<br>You can directly start executing the main blocks.

Instructions for the second notebook-
<br>There are no pre-requisite code blocks to run, you can start executing the code related to the tasks directly.

## Dependency Libraries Used and Approach followed:

### 1)  For word similarity tasks -
  **Genism:** used for working with Word2Vec embeddings
<br> **pandas:** for handling data
<br> **Natural Language Toolkit (nltk):**  to access the brown dataset and punkt using its resources
<br> **Numpy:** for numerical operations
<br> **Scipy:** for technical computing

### Approach
1.Load pre-trained Word2Vec model and SimLex-999 dataset.
<br>2.Define a function (calculate_similarity) to calculate similarity between words using Word2Vec embeddings.
<br>3.Tokenize and preprocess word pairs.
<br>4.Calculate word similarity using the defined function.
<br>5.Handle NaN values in the dataset.
<br>6.If the 'SimLex999' column is not constant, calculate and print Pearson and Spearman correlation coefficients.

### 2) For Phrase similarity task -
  <br> **PyTorch:** for building and training neural networks
   <br>**Genism:** used for working with Word2Vec embeddings
<br> **pandas:** for handling data
<br> **Natural Language Toolkit (nltk):**  to access punkt using its resources
<br> **Numpy:** for numerical operations
<br> **scikit-learn:** for machine learning related tasks
<br> **datasets:** for loading the hugging face dataset

### Approach

1.Load pre-trained Word2Vec model and the phrase similarity dataset.
<br>2.Handle missing values and outliers in the dataset.
<br>3.Define functions to calculate similarity between words using Word2Vec embeddings.
<br>4.Aggregate word representations into a phrase representation.
<br>5.Convert datasets to PyTorch tensors for training and testing.
<br>6.Create a simple neural network (PhraseSimilarityNN) for predicting phrase similarity.
<br>7.Train the neural network using PyTorch's Binary cross entropy with logitsloss and Adam optimizer.
<br>8.Evaluate the model on the validation set during training.
<br>9.Test the trained model on the test set and report accuracy.

### 3) For Sentence similarity task -
<br> **Spacy:** for loading the pre-trained model
<br> **scikit-learn:** for machine learning related tasks
<br> **datasets:** for loading the hugging face dataset
   
### Approach

1.Load SpaCy models for English (en_core_web_lg and en_core_web_md).
<br>2.Load the PAWS-X dataset using Hugging Face Datasets.
<br>3.Extract features and labels for training, development, and test sets.
<br>4.Calculate average similarity scores for each sentence pair using SpaCy.
<br>5.Combine original sentences for training logistic regression model.
<br>6.Train a logistic regression model using scikit-learn's TfidfVectorizer and LogisticRegression.
<br>7.Make predictions on the development and test sets using the trained model.
<br>8.Evaluate the model's accuracy on the development and test sets.

### 4) Bonus task-
<br> **transformers:** for loading BERT
<br> **pytorch:** to build and train neural networks
<br> **datasets:** for loading the hugging face dataset

### Approach
1.Install and load the necessary libraries: Transformers, Datasets, and PyTorch.
<br>2.Load the BERT tokenizer and pre-trained model for sequence classification.
<br>3.Load the PAWS-X dataset using Hugging Face Datasets.
<br>4.Implement a custom dataset class (SentenceSimilarityDataset) or (PhraseSimilarityDataset) for tokenizing and preparing the data.
<br>5.Split the dataset into training, development, and test sets.
<br>6.Train a BERT-based model for sentence similarity classification using PyTorch.
<br>7.Define the optimizer and configure the training loop.
<br>8.Train the model and evaluate its performance on the development set during training.
<br>9.Test the trained model on the test set and report accuracy.
