from dataclasses import fields
from typing import List
import openai
import configparser
import pandas as pd
from string import Template
from constants import (
    IMPROVE_TEMPLATE,
    PERSONA_TEMPLATE,
    SEPARATOR,
    PRIMING_SEPARATOR,
    CSV_MSG_SEPARATOR,
    MAX_TOKENS_GPT3,
    MAX_TOKENS_GPT4,
    IN_CONTEXT_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    INDUCE_TEMPLATE,
    TOPIC_TEMPLATE,
)
from dataclss import ChatResponse, CompletionResponse, DfDict, Message
from utils import msg_to_dicts, num_tokens_from_messages


class Gpt:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(".env")
        openai.api_key = config["keys"]["OPENAI_API_KEY"]
        openai.organization = config["keys"]["OPENAI_ORG_KEY"]

        self.model = "text-davinci-003"
        self.chat_model = "gpt-3.5-turbo"
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

    def get_config(self, use_chat_model: bool = False):
        return {
            "model": self.model if not use_chat_model else self.chat_model,
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
        )  # type: ignore

    def chat_completion(self, messages) -> ChatResponse:
        msg_dicts = msg_to_dicts(messages)
        num_tokens = num_tokens_from_messages(messages, self.model)
        return openai.ChatCompletion.create(
            model=self.chat_model,
            messages=msg_dicts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens - num_tokens,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
        )  # type: ignore

    # TODO: Add evaluation scores
    def to_df_dict(
        self,
        prompt_template: Template,
        response,
        examples: List[List[str]] = [[]],
        num_examples: int = 0,
        text="",
    ) -> DfDict:
        if isinstance(response, CompletionResponse):
            return DfDict(
                prompt_template.template,
                examples,
                num_examples,
                text,
                response.choices[0].text,
                response.choices[0].finish_reason,
            )
        msg_text = ""
        for m in response:
            msg_text += m.role + ": " + m.content + CSV_MSG_SEPARATOR

        return DfDict(
            prompt_template.template, examples, num_examples, text, msg_text, ""
        )

    def dict_to_df(self, info_dict: DfDict):
        conf = self.get_config()
        df = pd.DataFrame(
            [[getattr(info_dict, field.name) for field in fields(info_dict)] + [conf]],
            columns=[*list(fields(info_dict)), "config"],
        )
        return df

    def save_df(self, info_dict: DfDict, path: str):
        df = self.dict_to_df(info_dict)
        df.to_csv(path, mode="a", index=False, header=False)

    # TODO: Add "example_user" and "example_assistant" to the prompts with examples
    def create_chat_messages(
        self,
        prompt: str,
        text: str,
        approach: str,
        reference_text: str = "",
        reference_summary: str = "",
    ) -> List[Message]:
        messages = []
        match approach:
            case "in_context":
                messages = [
                    Message(
                        "user",
                        "I want you to summarize a text. Here are some representative examples of how to summarize a text.",
                    )
                ]
                examples = prompt.split(PRIMING_SEPARATOR)
                for example in examples:
                    messages.append(Message(example, "user"))
                messages.append(
                    Message(
                        "user",
                        "Now summarize the following text:"
                        + SEPARATOR
                        + text
                        + SEPARATOR,
                    )
                )

            case "follow_up_questions":
                messages = [
                    Message(
                        "user",
                        "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise.",
                    ),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                    Message(
                        "assistant",
                        "Before summarizing, I will ask two questions about the text.",
                    ),
                ]

            case "persona":
                messages = [
                    Message("system", prompt),
                    Message(
                        "user",
                        "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise.",
                    ),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                ]

            case "improve":
                messages = [
                    Message(
                        "user",
                        "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise.",
                    ),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                ]

            case "important_parts":
                messages = [
                    Message("user", "Find all important parts of the following text."),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                ]

            case "summarize_important_parts":
                messages = [
                    Message(
                        "user",
                        "Here is a text and its important parts. Summarize the text into three subheadings with three corresponding bullet points.",
                    ),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                    Message(
                        "user", "Important parts:" + SEPARATOR + prompt + SEPARATOR
                    ),
                ]

            case "induce":
                messages = [
                    Message("user", prompt),
                ]

            case "template":
                messages = [
                    Message(
                        "user",
                        "I am going to provide you a template for your output. Everything in all caps is a placeholder. The prompts must result in output summaries that follow the template format. Template:"
                        + SEPARATOR
                        + prompt
                        + SEPARATOR,
                    ),
                    Message(
                        "user",
                        "Summarize the following text:"
                        + SEPARATOR
                        + text
                        + SEPARATOR
                        + "Be concise.",
                    ),
                ]

            case "template_with_examples":
                messages = [
                    Message(
                        "user",
                        "I am going to provide you a template for your output. Everything in all caps is a placeholder. The prompts must result in output summaries that follow the template format. Template:"
                        + SEPARATOR
                        + prompt
                        + SEPARATOR,
                    ),
                    Message(
                        "user",
                        "A sample interaction after the prompt was provided is shown below.\n"
                        + SEPARATOR
                        + "\nSummarize the text into three subheadings with three corresponding bullet points. Be concise."
                        + "\nText: "
                        + reference_text
                        + "\nOutput: "
                        + reference_summary
                        + SEPARATOR,
                    ),
                    Message(
                        "user",
                        "Summarize the following text:"
                        + SEPARATOR
                        + text
                        + SEPARATOR
                        + "Be concise.",
                    ),
                ]

            # TODO: Update this
            case "kitchen-sink":
                messages = [
                    Message("system", prompt),
                    Message(
                        "user",
                        "I want you to summarize a text into three subheadings with three corresponding bullet points.",
                    ),
                    # Message("user" , "Text: " + text),
                    Message(
                        "user",
                        "I want you to summarize a text for me. Here are some representative examples of how to summarize a text.",
                    ),
                ]
                examples = prompt.split(SEPARATOR)
                for example in examples:
                    messages.append(Message(SEPARATOR + example + SEPARATOR, "user"))
                messages.append(
                    Message(
                        "assistant",
                        "Before summarizing, I will ask two questions about the text.",
                    )
                )

        return messages

    def follow_up_completion(self, text: str) -> List[Message]:
        messages = self.create_chat_messages("", text, "follow_up_questions")
        final = "Now summarize the text into three subheadings with three corresponding bullet points. Be concise."
        role = "user"
        for _ in range(3):
            response = self.chat_completion(messages)
            message = response.choices[0].message
            messages.append(Message(role, message.content))
            role = "user" if role == "assistant" else "assistant"

        messages.append(Message("user", final))
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return messages[4:]

    def follow_up_summarization(self, text) -> DfDict:
        messages = self.follow_up_completion(text)
        return self.to_df_dict(FOLLOW_UP_TEMPLATE, messages, [[]], 0, text)

    def current_sum(self, text: str) -> CompletionResponse:
        current_strategy = "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n"
        response = self.completion(current_strategy + text)
        return response

    def zero_shot_completion(self, text: str):
        # Heading
        heading_response = self.completion(
            "Text:" + SEPARATOR + text + SEPARATOR + "\n\nThree subheadings:",
        )
        # Bullet points
        bullet_response = self.completion(
            "Text:"
            + SEPARATOR
            + text
            + SEPARATOR
            + "\n\nSubheadings:"
            + SEPARATOR
            + heading_response.choices[0].text
            + SEPARATOR
            + "\n\nBullet points:"
        )
        return heading_response, bullet_response

    def topic_completion(self, text: str, topic: str) -> CompletionResponse:
        prompt = "Summarize the following text into three subheadings with three corresponding bullet points. Be concise."
        prompt += "\n\nTopic:" + SEPARATOR + topic + SEPARATOR
        prompt += "\n\nText:" + SEPARATOR + text + SEPARATOR
        prompt += "\n\nSummary:"

        return self.completion(prompt)

    def in_context_completion(
        self, inputs: List[str], outputs: List[str], text: str, useChat: bool
    ) -> CompletionResponse | ChatResponse:
        prompt = ""
        for i, o in zip(inputs, outputs):
            prompt += "Input: " + i + "\nOutput:" + o + PRIMING_SEPARATOR
        prompt += "Input:" + text + "\n" + "Output:"

        response = {}
        if not useChat:
            response = self.completion(prompt)
        else:
            response = self.chat_completion(
                self.create_chat_messages(prompt, text, "in_context")
            )

        return response

    def in_context_prediction(
        self, examples: List[List[str]], text: str, num_examples: int, useChat=False
    ) -> DfDict:
        res = self.in_context_completion(
            examples[0][:num_examples], examples[1][:num_examples], text, useChat
        )
        return self.to_df_dict(IN_CONTEXT_TEMPLATE, res, examples, num_examples, text)

    def induce_completion(self, inputs: List[str], outputs: List[str], useChat=False):
        context = "I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs.\n"
        if not useChat:
            prompt = ""
            prompt_examples = ""
            for i, o in zip(inputs, outputs):
                prompt_examples += "Input:" + i + "\nOutput:" + o + PRIMING_SEPARATOR
            before_pred = "The instruction was"

            prompt += context + prompt_examples + before_pred

            return self.completion(prompt)

        else:
            messages = self.create_chat_messages(context, "", "induce")
            for i, o in zip(inputs, outputs):
                messages.append(
                    Message("user", "Input:" + i + "\nOutput:" + o + PRIMING_SEPARATOR)
                )
            messages.append(Message("user", "What was the instruction?"))

            return self.chat_completion(messages)

    def induce_instruction(
        self, examples: List[List[str]], num_examples: int
    ) -> DfDict:
        res = self.induce_completion(
            examples[0][:num_examples], examples[1][:num_examples]
        )
        return self.to_df_dict(INDUCE_TEMPLATE, res, examples, num_examples)

    def generate_persona_context(self, topics):
        topic_list = topics.split(",")
        topic_str = ""
        for t in topic_list:
            topic_str += t + ", "
        return "You are an expert in the following topics: ###\n" + topic_str + ".\n###"

    def persona_completion(self, text: str, persona_context_setter: str, useChat: bool):
        strings = [
            persona_context_setter,
            "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.",
            "Text: ###\n" + text + "\n###",
            "Summary:",
        ]

        if not useChat:
            prompt = ""
            for s in strings:
                prompt += s

            return self.completion(prompt)

        else:
            messages = self.create_chat_messages(
                persona_context_setter, text, "persona"
            )
            response = self.chat_completion(messages)
            messages[-1].content = "This is the text I want you to summarize: *text*"
            messages.append(response.choices[0].message)

            return messages

    def persona_summarization(
        self,
        text: str,
        persona_context: str,
        useChat: bool = False,
        num_predictions: int = 1,
    ):
        info_dict = self.to_df_dict(
            PERSONA_TEMPLATE,
            self.persona_completion(text, persona_context, useChat),
            text=text,
        )
        return info_dict

    def improve_completion(self, text: str, useChat: bool = True):
        prompt = "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.\n"
        prompt += "Text: ###\n" + text + SEPARATOR
        prompt += "Summary:"

        if not useChat:
            response = self.completion(prompt)
            summary = response.choices[0].text
            prompt = "Here is a text and its summary. Please improve the summary.\n"
            prompt += "Text: ###\n" + text + SEPARATOR
            prompt += "Summary: ###\n" + summary + SEPARATOR
            return self.completion(prompt)

        else:
            messages = self.create_chat_messages(prompt, text, "improve")
            response_msg = self.chat_completion(messages).choices[0].message
            messages.append(Message(response_msg.role, response_msg.content))
            messages.append(
                Message(
                    "user",
                    "I am not satisfied with the summary. Please improve it using three subheadings with three corresponding bullet points. Be concise.",
                )
            )
            response_msg = self.chat_completion(messages).choices[0].message
            messages.append(Message(response_msg.role, response_msg.content))

            return messages

    def improve_summarization(
        self, text: str, useChat: bool = False, num_predictions: int = 1
    ):
        info_dict = self.to_df_dict(
            IMPROVE_TEMPLATE, self.improve_completion(text, useChat), text=text
        )
        return info_dict

    def important_parts_completion(self, text: str, useChat: bool = False):
        if not useChat:
            prompt = "Find all important parts of the following text.\n"
            prompt += "Text: ###\n" + text + SEPARATOR
            prompt += "Important parts:"
            important_parts = self.completion(prompt).choices[0].text

            prompt = "Here is a text and its important parts. Please improve the important parts.\n"
            prompt += "Text:" + SEPARATOR + text + SEPARATOR + "\n\n"
            prompt += "Important parts:" + SEPARATOR + important_parts + SEPARATOR
            important_parts = self.completion(prompt).choices[0].text

            prompt = "Here is a text and its important parts. Summarize the content into a three subheadings with three corresponding bullet points.\n"
            prompt += "Text:" + SEPARATOR + text + SEPARATOR + "\n\n"
            prompt += "Important parts:" + SEPARATOR + important_parts + SEPARATOR
            response = self.completion(prompt).choices[0].text

            return response

        else:
            messages = self.create_chat_messages("", text, "important_parts")
            messages.append(self.chat_completion(messages).choices[0].message)
            messages.append(
                Message(
                    "user",
                    "I am not satisfied with the important parts. Please improve them.",
                )
            )
            important_parts = self.chat_completion(messages).choices[0].message

            final_messages = self.create_chat_messages(
                important_parts.content, text, "summarize_important_parts"
            )

            return self.chat_completion(final_messages).choices[0].message

    def template_completion(
        self,
        text: str,
        template: str,
        useChat: bool = False,
        reference_text: str = "",
        reference_summary: str = "",
    ):
        template = "\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET"
        if not useChat:
            prompt = "I am going to provide you a template for your output. Everything in all caps is a placeholder. The prompts must result in output summaries that follow the template format. Template:"
            prompt += SEPARATOR + template + SEPARATOR
            prompt += "\n\nA sample interaction after the prompt was provided is shown below.\n"
            prompt += (
                SEPARATOR
                + "User: Summarize the text into three subheadings with three corresponding bullet points. Be concise."
            )
            prompt += "\nText: " + reference_text
            prompt += "\nOutput: " + reference_summary + SEPARATOR
            prompt += "\n\nNow, please summarize the following text into three subheadings with three corresponding bullet points. Be concise."
            prompt += "\nText: " + SEPARATOR + text + SEPARATOR
            return self.completion(prompt).choices[0].text

        else:
            messages = self.create_chat_messages(template, text, "template")
            return self.chat_completion(messages).choices[0].message

    def repeat_completion(self, text: str, useChat: bool = False):
        pass
