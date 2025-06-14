import cohere
import os
from dotenv import load_dotenv
##this ensures 
load_dotenv()

API_KEY = os.getenv("SECRET_API_KEY")
if API_KEY is None:
    raise RuntimeError("SECRET_API_KEY not set in environment")


model = cohere.ClientV2(API_KEY) ##trying to get me api key 


def get_response(query, context=""):  ##creating a function for the 
    messages = [{
        "role": "system",
        "content": (
            "You are an AI assistant. Use the context provided by the user to give the user a concise answer to their prompt. "
            "If the answer isn't present do not make it up, rather, inform the user that you do not know the answer"
        )
    }]
    if context:  # only add the context when non-empty
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": query})

    response = model.chat(
        model="command-a-03-2025",
        messages=messages
    )
    return response.message.content[0].text.strip()





## testing model:

get_response("How are you?", "Just reply")