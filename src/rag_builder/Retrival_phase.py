from .LLM_Inference import get_response
from .Ingesting_phase import Doc_Vectorizer ##importing the doc_vectorizer class 
import requests 


dv=Doc_Vectorizer() ##instantiting the Doc_vectorizer class

#the code below should clear each storage variable after each session so previous docs don't interupt with new inputs
def reset_database(): 
    dv.vectorized_docs.clear()
    dv.original_docs.clear()  
    dv.vectors= None

# this should take the file type using the file name, e.g if you input a .txt file it should be able to infer that you are using a txt file
def initialize(file_name):
    file_type= file_name.split(".")[-1] 
    return dv.process_and_add_documents(file_path=file_name, file_type=file_type)


def chat(user_query, is_debug= False):
    original_best_match, processed_best_match= dv.find_best_matches(user_query)
    context= "\n\n".join(original_best_match[0])

    if is_debug: ##this is ensure that our model can give us some context when we aren't able to get a response
        print(f"Context: {context}")

    resp= get_response(user_query, context)
    return resp


