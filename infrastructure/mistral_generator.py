import os
from mistralai import Mistral
from domain import IGenerator
from dotenv import load_dotenv
load_dotenv()
class MistralGenerator(IGenerator):
    def __init__(self, model_name : str="mistral-medium-latest"):
        self.model_name = model_name
        self.api_key=os.getenv("MISTRAL_API_KEY")
        self.client=Mistral(api_key=self.api_key)
    def generate(self, formatted_prompt: str) -> str:
        chat_response= self.client.chat.complete(
            model= self.model_name,
            messages = [{"role": "system", "content": "You are GymRat, a helpful assistant specialized in fitness, "
                                                      "workout routines, muscle growth, fat loss, and nutrition.\n\n"
                                                      "Below are previously answered questions and their expert responses. "
                                                      "Use them as context to answer the new question accurately.\n\n"},
                        {"role": "user", "content": formatted_prompt}])

        return chat_response.choices[0].message.content



