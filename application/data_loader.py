import re
from datasets import load_dataset

def parse_alpaca_format(sample):
    text = sample['text'].strip()
    text = re.sub(r"^<s>|\s*</s>\s*", "", text)
    match = re.search(r"\[INST\](.*?)\[/INST\](.*)", text, re.DOTALL)
    if not match:
        return {'instruction': '', 'response': ''}
    instruction_raw, response = match.groups()
    instruction = re.sub(r"<<SYS>>.*?<</SYS>>", "", instruction_raw)
    return {'instruction': instruction.strip(), 'response': response.strip()}

def load_and_prepare_dataset():
    dataset = load_dataset("onurSakar/GYM-Exercise")["train"]
    return dataset.map(parse_alpaca_format)
