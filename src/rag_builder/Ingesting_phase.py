##importing relevant dependencies 

import os 
import nltk ##for data preprocessing 
from PyPDF2 import PdfReader  ##for handling reading of the PDFs
from bs4 import BeautifulSoup ##for web scrapping 
from nltk.stem import PorterStemmer ##for stemming
from nltk.tokenize import sent_tokenize ##for tokenizing our inputs
nltk.download("punkt_tab") ##for handling punctuations 
from sklearn.feature_extraction.text import TfidfVectorizer


# Step 1: Data Pre-processing 

##Stemming of the inputting data 

stemmer= PorterStemmer() ##initiallizng our stemmer

## building logic for stemming and ddata processing

Chunk_size= 999999 ##declaring my defualt chunk size in case the user doesn't specify 

def process_text(text, Chunk_size= Chunk_size):
    sentences = sent_tokenize(text) ##tokenizing any text we recieve 
    ##I will be creating three variables 

    original_text= [] #this is for storing the original text we got from the user for easy retrival
    processed_text= [] #this will store our processed text after the original has been passed through this function
    segments= ""  ##this is for storing the chunked up data 
    ##I will explain this code in a string below
    for text in sentences:
        if len(segments) + len(text) > Chunk_size:
            original_text.append(segments)
            processed_text.append(" ".join([stemmer.stem(word) for word in segments.split()]))
            segments = text
        segments += " " + text

        # Split text into chunks of at most Chunk_size:
        # when adding the next sentence would overflow the chunk,
        # flush the current segment (unchanged and stemmed) to the outputs
        # and start a new segment.


        ##Handling the last sequence 

    if segments:
        original_text.append(segments)
        processed_text.append(" ". join([stemmer.stem(word) for word in segments.split()]))

    return original_text, processed_text


# Step 2: Ingesting the file : We will allow the file take in a PDF, text, or HTML file

##the initial code consisted of three functions but I refactored them into a single class 


class DocumentLoader:

    def __init__(self, file_path):
        self.file_path= file_path

## a method for loading and reading PDFs
    def load_pdf(self):
        with open(self.file_path, "rb") as f: ## We are using 'rb' since PDFs are compressed so we use rb to read it instead of reading it raw as we would with text files 
            reader= PdfReader(f)
            text= ""
            for x in reader.pages:
                text += x.extract_text()
            return process_text(text)

## a method for handling txt files 

    def load_text(self):
        with open(self.file_path, "r") as f: ## we are using r to have it read the raw text 
            text= f.read()
            return process_text(text)
    
## A method for handling html files 
    def load_html(self):
        with open(self.file_path, "r") as f:
            data= BeautifulSoup(f, "html.parser")
            text= data.get_text()
            return process_text(text)


# Step 3  Vectorization and Similarity Searching  I am creating one class for the vectorizer 



## a class for handling adding documents 

class Doc_Vectorizer:

    def __init__(self):

        self.vectorizer= TfidfVectorizer()

        self.vectorized_docs= []

        self.original_docs= []

        self.vectors= None 


    def add_documents(self, text):
        self.vectorized_docs.extend(text)
        self.vectors= self.vectorizer.fit_transform(self.vectorized_docs)
        return self.vectors 

    def process_and_add_documents(self, file_path, file_type): ##this is a method for handling multiple 
        file_type= file_type ##this should help handle various casing inputs of the variables 
        doc_loader= DocumentLoader(file_path) ##intiating the documentloader class
        if file_type == "pdf":
            original_data, processed_data= doc_loader.load_pdf()
        elif file_type== "txt":
            original_data, processed_data= doc_loader.load_text()
        elif file_type== "html":
            original_data, processed_data= doc_loader.load_html()
        else: 
            raise TypeError("You provided an incorrect file type")
        self.original_docs.append(original_data)
        self.vectors= self.add_documents(processed_data)

        return self.vectors 
        
    def find_best_matches(self, query, k=3):
        process_query = process_text(query)[1]
        query_vector= self.vectorizer.transform(process_query)
        similarity= (query_vector * self.vectors.T).toarray() ##we are using the dot product to find out the similarity score
        best_match= similarity.argsort()[0][-k:][::-1] ##this code should sort the output, and then grab the top k (3) and then reverse them
        return [self.original_docs[i] for i in best_match], [self.vectorized_docs[i] for i in best_match]


#Construction of LLM and Outputs 

##For the LLM I will be using one from Cohere since I cannot 



