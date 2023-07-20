import json
import time
import openai
import pandas as pd

from datasets import ClassLabel, Dataset, Features, Value, concatenate_datasets
from langchain import PromptTemplate, OpenAI
from langchain.chat_models import AzureChatOpenAI

from utils import load_content


class GPTDataAugmentation:
    def __init__(self, data, open_api_key):
        self.data = data
        self.batch_call_size = 5
        openai.api_key = open_api_key 

    def augment(self):

        new_examples = []
        original_sentences = [example["text"] for example in self.data]

        # Call OpenAI api in batch to save time and cost
        chunk_size = self.batch_call_size
        batched_response = []
        for i in range(len(original_sentences) // chunk_size + 1):
            batch = original_sentences[i*chunk_size: min((i+1)*chunk_size, len(original_sentences))]
            # Call OpenAI with prompt template
            response = self.generate_samples(batch)
            batched_response.append(response)

        # Loop the augement sentences, flattening the response
        batched_response = [item for sublist in batched_response for item in sublist]
        for example, response in zip(self.data, batched_response):
            for output in ["o1","o2","o3"]:                
                new_example = example.copy()
                new_example["text"] = response[output]
                new_examples.append(new_example)

        # data schema
        class_label = ClassLabel(names=['neg', 'pos'])
        schema = Features(
            {
                'text': Value('string'),  # Example feature
                'label': class_label,  # Assign the ClassLabel to the 'label' feature
            }
        )
        augmented_dataset = Dataset.from_pandas(
            pd.DataFrame(new_examples), features=schema
        )
        # Combine the new examples into an extended dataset
        extended_dataset = concatenate_datasets([self.data, augmented_dataset])

        return extended_dataset
        
    def clean_response_text(self, response_text):
        return response_text.replace("\n\n", "\n")

    def call_openai_and_get_response(self, messages, max_retries=3, retry_interval=5):
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                reply = {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content,
                }
                return reply
            except Exception as ex:
                # print(f"Error: {ex}")
                time.sleep(retry_interval)
                continue
        error_msg = (
            "The server had an ERROR processing your request. Please retry your request."
        )
        raise Exception(error_msg)

    def generate_samples(self, sentences):

        template = load_content(path="./data/prompts/prompt.txt")

        # prepare user prompt
        prompt = PromptTemplate.from_template(template)
        input_text = "\n".join([f"[{index}] {sentence}" for index, sentence in enumerate(sentences)])
        _input = prompt.format(input_text=input_text)

        messages = []
        messages.append(
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            }
        )
        messages.append({"role": "user", "content": _input})
        reply = self.call_openai_and_get_response(messages)
        result = json.loads(reply['content'])
        return result
