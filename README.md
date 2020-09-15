# Course-Recommender-Chatbot

# Installation Instructions
Following Libraries needs to be installed:-
1) Pandas
2) Numpy
3) NLTK
4) Tensorflow

# How is dataset created
1) Scrap data from various sources like coursera , edx through their APIs
2) Convert the scrapped data into appropriate json form that chatbot understands. 
3) Create a unique tagging system to cluster common courses.
4) The JSON File contains tags(unique tags assigned to scrapped data), patterns (how a user is expected to ask the chatbot) (This is created by automatically adding following as intents - I want to learn {tag} , I am interested in studying {tag} ) , and then responses which are basically the courses for the tag.

The dataset is also made public at Kaggle:- https://www.kaggle.com/nisargbshah/dataset-for-a-chatbot-to-recommend-courses

# Data Preparation for training
1) Extract words from each pattern & create a bag of words. Sort the bag of words lexicographically(dictionary order) , Tokenization & Stemming done on words.
2) Extract the tag as a label(This is what we want as output from our NN model so that we can use it for extracting the courses). Sort the labels as well.
3) Preparing the training dataset :- 
Dimension of dataset :- nxm where n:- No of tags in dataset , m:- No of words in bag of words.
Dimension of output :- nx1 
For every tag training data vector is a 0-1 vector of size m, 0 represents word present at index i in bag of words is not present in the corresponding pattern & 1 represents the presence of that word.
The output vector is also similar.

# Training DL model
1) Train a DL model to perform the task.
2) DL model consists of fully connected & softmax layers.
3) Training done for 1000 epochs and final model is saved in finalized_model.tflearn

# Predicting
1) User inputs a sentence like (I want to learn Python)
2) Create a bag of words vector(0-1 vector) & pass it to DL model to identify the tag.
3) From the tag retrive the responses from the json , beautify it & output it to user

# UI for easier usage ( To be done :- Web based/Desktop UI)
1) An option for user to enter what they want to learn.
2) Finally user can download a pdf file that consists recommendations of courses
