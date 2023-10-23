from langchain.llms.openai import OpenAI

openai_api_key = ''

INITIAL_PROMPT = ''


class LangNav:
    def __init__(self, llm_name: str):

        if llm_name == 'gpt':
            llm = OpenAI(openai_api_key=openai_api_key)
            
        

