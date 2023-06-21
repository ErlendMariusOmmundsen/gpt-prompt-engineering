from dataclasses import asdict
import re
from typing import List
import tiktoken
import pandas as pd

from dataclss import Message


def get_examples() -> List[List[str]]:
    # use pandas to read in examples from data/manual_summaries.csv and return them as a list containing two lists of strings, one for each column
    df = pd.read_csv("data/manual_summaries.csv")
    return [df["transcript"].tolist(), df["summary"].tolist()]


def msg_to_dicts(messages: List[Message]) -> List[dict]:
    return [asdict(m) for m in messages]


def messages_to_string(messages: List[Message]) -> str:
    result = ""
    for message in messages:
        result += message.role + ":\n" + message.content + "\n\n"
    return result


def remove_prefix_words(string: str) -> str:
    for word in [
        "headings",
        "heading",
        "subheadings",
        "subheading",
        "sub-heading",
        "bullet-points",
        "bullet-point",
        "bullets",
        "bullet",
        "summary",
        "i.",
        "i:",
        "ii.",
        "ii:",
        "iii.",
        "iii:",
    ]:
        if string.lower().startswith(word.lower()):
            return string[len(word) :]
    return string


def remove_prefix_str(string: str) -> str:
    string = remove_prefix_words(string).replace("\n", "")
    string = string.strip(" -.*:")
    try:
        if string[0].isdigit() and (string[1] == "." or string[1] == ":"):
            string = string[string.find(".") + 2 :]
            return string.strip(" -.*:")
        else:
            return string
    except:
        return string


def remove_suffix_str(string: str) -> str:
    if string.endswith(": "):
        return string[:-2]
    elif string.endswith(":"):
        return string[:-1]
    else:
        return string


def remove_empty_lines(string: str) -> str:
    return re.sub(r"\s*\n\s*\n\s*", "\n", string)


def clean_prediction(prediction: str) -> str:
    prediction = remove_empty_lines(prediction)
    prediction_splits = prediction.split("\n")
    for i in range(len(prediction_splits)):
        prediction_splits[i] = remove_prefix_str(prediction_splits[i])
        prediction_splits[i] = remove_suffix_str(prediction_splits[i])
    clean_splits = []
    for line in prediction_splits:
        if len(line) > 2:
            clean_splits.append(line)
    prediction = "\n".join(clean_splits)
    prediction = remove_empty_lines(prediction)
    return prediction


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_examples(examples, model="gpt-4") -> int:
    num_tokens = 0
    for example_list in examples:
        for example in example_list:
            num_tokens += num_tokens_from_string(example, model) + 4
    return num_tokens


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
def num_tokens_from_messages(messages, model="gpt-4") -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
