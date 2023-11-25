import streamlit as st
import pinecone      
import openai
openai.api_key = "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"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pinecone.init(
	api_key='ad3e7100-9bd0-444f-a8fd-01e274752c65',
	environment='gcp-starter'
)
index = pinecone.Index('ind')
def get_conversation_string():
    conversation_string = ""
    print(st.session_state['responses'])

    for i in range(len(st.session_state['responses'])-1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"

    return conversation_string


def query_refiner(conversation_log, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation_log}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def find_match(input_query):
    input_embed = model.encode(input_query).tolist()
    result = index.query(input_embed, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']



