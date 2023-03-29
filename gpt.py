from dataclasses import fields
from typing import List
import openai
import configparser
import pandas as pd
from string import Template
from constants import PERSONA_TEMPLATE, SEPARATOR, CSV_MSG_SEPARATOR, MAX_TOKENS_GPT3, MAX_TOKENS_GPT4, IN_CONTEXT_TEMPLATE, FOLLOW_UP_TEMPLATE, INDUCE_TEMPLATE, TOPIC_TEMPLATE
from dataclss import ChatResponse, CompletionResponse, DfDict, Message
from utils import msg_to_dicts, num_tokens_from_string

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


    def completion(self, prompt: str) -> CompletionResponse:
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
        msg_dicts = msg_to_dicts(messages)
        a = 40
        for msg in messages:
            a += num_tokens_from_string(msg.content, self.model)
        return openai.ChatCompletion.create(
            model=self.model,
            messages=msg_dicts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens - a,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
        ) # type: ignore


    # TODO: Add evaluation scores
    def to_df_dict(self, prompt_template:Template, response, examples: List[List[str]]=[[]], num_examples: int=0, text="") -> DfDict:
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
        
    def dict_to_df(self, info_dict:DfDict):
        conf = self.get_config()
        df = pd.DataFrame(
            [[getattr(info_dict, field.name) for field in fields(info_dict)] + [conf]],
            columns=[*list(fields(info_dict)), "config"],
        )
        return df

    def save_df(self, info_dict:DfDict, path:str):
        df = self.dict_to_df(info_dict)
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
                    Message("user", "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise."),
                    Message("user" , "Text:###\n" + text + "\n###"),
                    Message("assistant", "Before summarizing, I will ask two questions about the text."),
                    ]

            case "persona":
                messages = [
                    Message("system", prompt),
                    Message("user", "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise."),
                    Message("user" , "Text:###\n" + text + "\n###"),
                    ]

            case "kitchen-sink":
                messages = [
                    Message("system", prompt),
                    Message("user", "I want you to summarize a text into three subheadings with three corresponding bullet points."),
                    # Message("user" , "Text: " + text),
                    Message("user", "I want you to summarize a text for me. Here are some representative examples of how to summarize a text.")
                ]
                examples = prompt.split(SEPARATOR)
                for example in examples:
                    messages.append(Message(SEPARATOR + example, "user"))
                messages.append(Message("assistant", "Before summarizing, I will ask two questions about the text."))
                    
        return messages


    def follow_up_prediction(self, text:str) -> List[Message]:
        messages = self.create_chat_messages("", text, "follow_up_questions")
        final = "Now summarize the text into three subheadings with three corresponding bullet points. Be concise."
        role = "user"
        for _ in range(3):
            response = self.chat_completion(messages) 
            message = response.choices[0].message
            messages.append(Message(role, message.content))
            role = "user" if role=="assistant" else "assistant" 
        
        messages.append(Message("user", final))
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return messages[4:]


    def follow_up_predictions(self, text) -> DfDict:
        temp = FOLLOW_UP_TEMPLATE
        messages = self.follow_up_prediction(text)
        return self.to_df_dict(temp, messages, [[]], 0, text)


    def follow_up_pipe(self, text):
        info_dict = self.follow_up_predictions(text)
        self.save_df(info_dict, "follow-up.csv")


    def current_sum(self, text: str) -> CompletionResponse:
        current_strategy = "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n"
        response = self.completion(current_strategy + text)
        return response


    def simple_sum(self, text: str):
        # Bullet points
        bullet_response = self.completion("Text: ###\n" + text + SEPARATOR + "Bullet points:")
        # Heading
        heading_response = self.completion(
            "Text: ###\n" + text + SEPARATOR + "Heading:",
        )

        return heading_response, bullet_response

    
    def topic_sum(self, text: str, topic:str) -> CompletionResponse:
        prompt = "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.\n"
        prompt += "Topic: ###\n" + topic + SEPARATOR
        prompt += "Text: ###\n" + text + SEPARATOR
        prompt += "Summary:"

        return self.completion(prompt)


    def topic_pipe(self, text, topic):
        info_dict = self.to_df_dict(TOPIC_TEMPLATE, self.topic_sum(text, topic), text=text)
        self.save_df(info_dict, "in-context.csv")
        

    def in_context_prediction(
        self, inputs: List[str], outputs: List[str], text: str, useChat: bool
    ) ->  CompletionResponse | ChatResponse :
        prompt = ""
        for i, o in zip(inputs, outputs):
            prompt += "Input: ###\n" + i + SEPARATOR + "Output: ###\n" + o + SEPARATOR
        prompt += "Input: ###\n" + text + SEPARATOR + "Output:"

        response = {}

        if not useChat:
            response = self.completion(prompt)
        else:
            response = self.chat_completion(
                self.create_chat_messages(prompt, text, "in_context")
            )

        return response


    def in_context_predictions(
        self, examples: List[List[str]], text: str, num_examples: int, useChat=False
    ) -> DfDict:
        # TODO: Select random num of examples?
        res = self.in_context_prediction(
            examples[0][:num_examples], examples[1][:num_examples], text, useChat
        )
        return self.to_df_dict(IN_CONTEXT_TEMPLATE, res, examples, num_examples, text)


    def in_context_pipe(self, examples: List[List[str]], text: str, num_examples: int, useChat=False):
        info_dict = self.in_context_predictions(examples, text, num_examples, useChat)
        self.save_df(info_dict, "in-context.csv")


    def induce_instruction(self, inputs: List[str], outputs: List[str]):
        prompt = ""
        context = (
            "I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs.\n"
        )
        prompt_examples = ""
        for i, o in zip(inputs, outputs):
            prompt_examples += "Input: ###\n" + i + SEPARATOR + "Output: ###\n" + o + SEPARATOR
        before_pred = "The instruction was"

        prompt += context + prompt_examples + before_pred

        return self.completion(prompt)


    def induce_instructions(self, examples: List[List[str]], num_examples: int)-> DfDict:
        # TODO: Select random num of examples?
        res = self.induce_instruction(examples[0][:num_examples], examples[1][:num_examples])
        return self.to_df_dict(INDUCE_TEMPLATE, res, examples, num_examples)


    def induce_pipe(self, examples: List[List[str]], num_examples: int):
        info_dict = self.induce_instructions(examples, num_examples)
        self.save_df(info_dict, "instruction-induction.csv")

    def generate_persona_context(self, topics):
        topic_list = topics.split(",")
        topic_str = ""
        for t in topic_list:
            topic_str += t + ", "
        return "You are an expert in the following topics: ###\n" + topic_str + ".\n###"

    def persona_summarize(self, text:str, persona_context_setter: str, useChat: bool):
        strings = [persona_context_setter, 
                   "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.",
                   "Text: ###\n" + text + '\n###',
                   "Summary:"]

        if not useChat:
            prompt = "" 
            for s in strings:
                prompt += s

            return self.completion(prompt)

        else:
            messages = self.create_chat_messages(persona_context_setter, text, "persona")
            response = self.chat_completion(messages)
            messages[-1].content = "This is the text I want you to summarize: *text*"
            messages.append(response.choices[0].message)

            
            return messages


    def persona_summarizations(self, text: str, persona_context: str, useChat:bool = False, num_predictions: int = 1):
        info_dict = self.to_df_dict(PERSONA_TEMPLATE, self.persona_summarize(text, persona_context, useChat), text=text)
        df = self.dict_to_df(info_dict)
        
        for _ in range(num_predictions-1):
            info_dict = self.to_df_dict(PERSONA_TEMPLATE, self.persona_summarize(text, persona_context, useChat))
            df = pd.concat([df, self.dict_to_df(info_dict)])
        return info_dict


    def persona_pipe(self, text: str, topic: str, useChat: bool=True):
        if not useChat:
            self.model = "text-davinci-003"
        persona_context = self.generate_persona_context(topic)
        info_dict = self.persona_summarizations(text, persona_context, useChat)
        self.save_df(info_dict, "persona.csv")




       
