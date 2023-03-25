from dataclasses import fields
from typing import List
import openai
import configparser
import pandas as pd
from string import Template
from constants import SEPARATOR, CSV_MSG_SEPARATOR, MAX_TOKENS_GPT3, MAX_TOKENS_GPT4
from dataclss import ChatResponse, CompletionResponse, DfDict, Message
from utils import msg_to_dicts

class gpt:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(".env")
        openai.api_key = config["keys"]["OPENAI_API_KEY"]
        openai.organization = config["keys"]["OPENAI_ORG_KEY"]

        self.model = "gpt-3.5-turbo"
        self.prompt = ""
        self.suffix = ""
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
        self.temperature = 1
        # An alternative to sampling with temperature, called nucleus sampling, where the model considers the
        #    results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        self.top_p = 1
        self.max_tokens = MAX_TOKENS_GPT3 if "3" in self.model else MAX_TOKENS_GPT4
        # How many completions to generate for each prompt.
        self.n = 1
        # Whether to stream back partial progress.
        self.stream = False
        # Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. If logprobs is
        #    5, the API will return a list of the 5 most likely tokens.
        self.log_probs = None
        # Up to 4 sequences where the API will stop generating further tokens.
        self.stop = None


    def get_config(self):
        return {
            "model": self.model,
            "suffix": self.suffix,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
        }


    def completion(self, prompt) -> CompletionResponse:
        return openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            suffix=self.suffix,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.n,
            stream=self.stream,
            logprobs=self.log_probs,
            stop=self.stop,
        ) # type: ignore


    def chat_completion(self, messages) -> ChatResponse:
        return openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
        ) # type: ignore


    # TODO: Add evaluation scores
    def to_df_dict(self, prompt_template:Template, response, examples: List[List[str]], num_examples: int, text="") -> DfDict:
        if isinstance(response, CompletionResponse):
            return DfDict(
                prompt_template.template, 
                examples, 
                num_examples, 
                text, 
                response.choices[0].text,
                response.choices[0].finish_reason
            )
        msg_text = ""
        for m in response:
            msg_text += m.role + ": " + m.content + CSV_MSG_SEPARATOR

        return DfDict(
            prompt_template.template, 
            examples, 
            num_examples, 
            text, 
            msg_text,
            ""
        )
        

    def save_df(self, info_dict:DfDict, path:str):
        conf = self.get_config()
        df = pd.DataFrame(
            [[getattr(info_dict, field.name) for field in fields(info_dict)] + [conf]],
            columns=[*list(fields(info_dict)), "config"],
        )
        df.to_csv(path, mode="a", index=False, header=False)


    def create_chat_messages(self, prompt: str, text: str, approach: str) -> List[Message]:
        messages = []
        match approach:
            case "in_context":
                messages = [
                    Message("user", "I want you to summarize a text for me. Here are some representative examples of how to summarize a text.")
                ]
                examples = prompt.split(SEPARATOR)
                for example in examples:
                    messages.append(Message(SEPARATOR + example, "user"))
                messages.append(Message("user", "Now summarize the following text: " + text))

            case "follow_up_questions":
                messages = [
                    Message("system", "You are an expert at summarizing text."),
                    Message("user", "I want you to summarize a text into three subheadings with three corresponding bullet points."),
                    Message("user" , "This is the text I want you to summarize: " + text),
                    Message("assistant", "Before summarizing, I will ask two questions about the text."),
                    ]
        return messages


    def follow_up_prediction(self, text:str) -> List[Message]:
        messages = self.create_chat_messages("", text, "follow_up_questions")
        final = "Now summarize the text into three subheadings with three corresponding bullet points. Be concise."
        role = "user"
        for i in range(3):
            response = self.chat_completion(msg_to_dicts(messages)) 
            message = response.choices[0].message
            messages.append(Message(role, message.content))
            role = "user" if role=="assistant" else "assistant" 
        
        messages.append(Message("user", final))
        response = self.chat_completion(msg_to_dicts(messages))
        messages.append(Message("assistant", response.choices[0].message.content))
        return messages[4:]


    def follow_up_predictions(self, text) -> DfDict:
        temp = Template("Chat messages with follow_up_questions")
        messages = self.follow_up_prediction(text)
        return self.to_df_dict(temp, messages, [[]], 0, text)


    def follow_up_pipe(self, text):
        info_dict = self.follow_up_predictions(text)
        self.save_df(info_dict, "follow-up.csv")


    def current_summarize(self, text: str) -> CompletionResponse:
        current_strategy = "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n"
        response = self.completion(current_strategy + text)
        return response


    def simple_sum(self, text: str):
        # Bullet points
        bullet_response = self.completion("Text: " + text + "\nBullet points:")
        # Heading
        heading_response = self.completion(
            "Text: " + text + "\nHeading:",
        )

        return heading_response, bullet_response


    def in_context_prediction(
        self, inputs: List[str], outputs: List[str], text: str, useChat: bool
    ) -> tuple[Template, CompletionResponse | ChatResponse] :
        prompt_examples = ""
        for i, o in zip(inputs, outputs):
            prompt_examples += "Input: " + i + "\nOutput: " + o + SEPARATOR
        prompt_examples += "Input: " + text + "\nOutput:"

        temp = Template("Input: ${text} \nOutput:")
        prompt = prompt_examples + "Input: " + text + "\nOutput:"

        response = {}

        if not useChat:
            response = self.completion(prompt)
        else:
            response = self.chat_completion(
                self.create_chat_messages(prompt, text, "in_context")
            )

        return temp, response


    def in_context_predictions(
        self, examples: List[List[str]], text: str, num_examples: int, useChat=False
    ) -> DfDict:
        # TODO: Select random num of examples?
        prompt_template, res = self.in_context_prediction(
            examples[0][:num_examples], examples[1][:num_examples], text, useChat
        )
        return self.to_df_dict(prompt_template, res, examples, num_examples, text)



    def in_context_pipe(self, examples, text, num_examples, useChat=False):
        info_dict = self.in_context_predictions(examples, text, num_examples, useChat)
        self.save_df(info_dict, "in-context.csv")


    def induce_instruction(self, inputs, outputs, num_examples):
        prompt = ""
        context = (
            "I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:"
            + SEPARATOR
        )
        prompt_examples = ""
        for i, o in zip(inputs, outputs):
            prompt_examples += "Input: " + i + "\nOutput: " + o + SEPARATOR
        before_pred = "The instruction was"

        prompt += context + prompt_examples + before_pred
        temp = Template("Context_setter *sep* example_pairs *sep* The instruction was")

        return temp, self.completion(prompt)


    def induce_instructions(self, examples, num_examples)-> DfDict:
        # TODO: Select random num of examples?
        prompt_template, res = self.induce_instruction(
            examples[0][:num_examples], examples[1][:num_examples], 2
        )
        return self.to_df_dict(prompt_template, res, examples, num_examples)


    def induce_pipe(self, examples, num_examples):
        info_dict = self.induce_instructions(examples, num_examples)
        self.save_df(info_dict, "instruction-induction.csv")




