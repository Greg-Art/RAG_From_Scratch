import cohere
import os
from dotenv import load_dotenv
##this ensures 
load_dotenv()

model = cohere.ClientV2(os.getenv("SECRET_API_KEY")) ##trying to get me api key 


def get_response(query, context): ##creating a function for the function

    messages= [{"role": "system", "content": "You are an AI assistant. Use the context provided by the user to give the user a conise answer to their prompt. If the answer isn't present"
    "do not make it up, rather, inform the user that you do not know the answer"}, 
               {"role": "system", "content": context},
               {"role": "user", "content": query}
    
               ]
    
    response= model.chat(
        model= "command-r-03-2024", ### I am using this model as it is good for RAG and it is not the most recent, so hopefully it isn't trianed on the doc 
        messages= messages
    )
    ##getting the message 
    return response.message.content[0].text.strip() 



