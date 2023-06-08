from dataclasses import fields
import os
from typing import List
import openai
import configparser
import pandas as pd
from string import Template
from constants import (
    BASELINE_TEMPLATE,
    IMPORTANT_PARTS_TEMPLATE,
    IMPROVE_TEMPLATE,
    PERSONA_TEMPLATE,
    REPEAT_TEMPLATE,
    SEPARATOR,
    PRIMING_SEPARATOR,
    CSV_MSG_SEPARATOR,
    MAX_TOKENS_GPT3,
    MAX_TOKENS_GPT4,
    IN_CONTEXT_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    INDUCE_TEMPLATE,
    TOPIC_TEMPLATE,
    ZERO_SHOT_TEMPLATE,
)
from dataclss import ChatResponse, CompletionResponse, DfDict, Message
from utils import (
    msg_to_dicts,
    num_tokens_from_messages,
    num_tokens_from_string,
    clean_prediction,
)


class Gpt:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(".env")
        openai.api_key = config["keys"]["OPENAI_API_KEY"]
        openai.organization = config["keys"]["OPENAI_ORG_KEY"]

        self.model = "text-davinci-003"
        self.chat_model = "gpt-4"
        self.prompt = ""
        self.suffix = ""
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
        self.temperature = 1
        # An alternative to sampling with temperature, called nucleus sampling, where the model considers the
        #    results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        self.top_p = 1
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
            "n": self.n,
            "stop": self.stop,
        }

    def completion(self, prompt: str) -> CompletionResponse:
        num_tokens = num_tokens_from_string(prompt, "text-davinci-003")
        print("\tNumber of tokens in prompt:", num_tokens)
        return openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            suffix=self.suffix,
            top_p=self.top_p,
            max_tokens=MAX_TOKENS_GPT3 - num_tokens - 3,  # magic number?
            n=self.n,
            stream=self.stream,
            logprobs=self.log_probs,
            stop=self.stop,
        )  # type: ignore

    def chat_completion(self, messages) -> ChatResponse:
        msg_dicts = msg_to_dicts(messages)
        num_tokens = num_tokens_from_messages(msg_dicts, self.chat_model)
        print("\tNumber of tokens in prompt:", num_tokens)
        if num_tokens > MAX_TOKENS_GPT4:
            raise ValueError(
                f"Too many tokens in prompt: {num_tokens} > {MAX_TOKENS_GPT4}"
            )
        return openai.ChatCompletion.create(
            model=self.chat_model,
            messages=msg_dicts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=MAX_TOKENS_GPT4 - num_tokens - 3,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
        )  # type: ignore

    # TODO: Add evaluation scores
    def to_df_dict(
        self,
        prompt_template: Template,
        response,
        prompt: str = "",
        examples: List[List[str]] = [[]],
        num_examples: int = 0,
        text="",
    ) -> DfDict:
        if response.object == "text_completion":
            return DfDict(
                prompt_template.template,
                examples,
                num_examples,
                text,
                prompt,
                response.choices[0].text,
                response.choices[0].finish_reason,
            )
        else:
            message = response.choices[0].message
            print("\tMessage:", message)
            msg_text = clean_prediction(message.content)
            return DfDict(
                prompt_template.template,
                examples,
                num_examples,
                text,
                prompt,
                msg_text,
                response.choices[0].finish_reason,
            )

    def dict_to_df(self, info_dict: DfDict, use_chat_model: bool = True):
        conf = self.get_config(use_chat_model)
        field_objects = fields(info_dict)
        df = pd.DataFrame(
            [[getattr(info_dict, field.name) for field in field_objects] + [conf]],
            columns=[[field.name for field in field_objects] + ["config"]],
        )
        return df

    def save_df(self, info_dict: DfDict, path: str, use_chat_model: bool = True):
        df = self.dict_to_df(info_dict, use_chat_model)
        if not os.path.isfile(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False, mode="a", header=False)

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

            case "topic":
                messages = [
                    Message(
                        "user",
                        "I want you to summarize a text into three subheadings with three corresponding bullet points. Be concise. "
                        + "The text is on the following topic(s): "
                        + prompt,
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

            case "briefness":
                messages = [
                    Message(
                        "user",
                        "Summarize the following text into three subheadings with three corresponding bullet points."
                        + SEPARATOR
                        + text
                        + SEPARATOR
                        + "\nThe bullet points and subheadings should be "
                        + prompt,
                    ),
                ]

            case "shorten_as_possible":
                shorten_modifier = "Shorten the answer as possible."
                messages = [
                    Message(
                        "user",
                        "Summarize the following text into three subheadings with three corresponding bullet points."
                        + SEPARATOR
                        + text
                        + SEPARATOR
                        + "\n"
                        + shorten_modifier,
                    )
                ]

            case "quality":
                messages = [
                    Message(
                        "user",
                        "Summarize the following text into three subheadings with three corresponding bullet points."
                        + SEPARATOR
                        + text
                        + SEPARATOR
                        + "\nThe bullet points and subheadings should be "
                        + prompt,
                    )
                ]

            case "repeat":
                messages = [
                    Message(
                        "user",
                        "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.",
                    ),
                    Message("user", "Text:" + SEPARATOR + text + SEPARATOR),
                    Message(
                        "user",
                        "Summarize the text into three subheadings with three corresponding bullet points. Be concise.",
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
        role = "user"
        for _ in range(3):
            response = self.chat_completion(messages)
            message = response.choices[0].message
            messages.append(Message(role, message.content))
            role = "user" if role == "assistant" else "assistant"

        final = "Now summarize the text into three subheadings with three corresponding bullet points. Be concise."
        messages.append(Message("user", final))

        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return messages[4:]

    def follow_up_summarization(self, text) -> DfDict:
        messages = self.follow_up_completion(text)
        return self.to_df_dict(FOLLOW_UP_TEMPLATE, messages, [[]], 0, text)

    def baseline_summarization(self, text: str) -> CompletionResponse:
        current_strategy = "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n"
        messages = Message("user", current_strategy + text)
        response = self.chat_completion([messages])
        return self.to_df_dict(BASELINE_TEMPLATE, response, messages, [[]], 0, text)

    def zero_shot_summarization(self, text: str, prompt: str) -> DfDict:
        messages = self.create_chat_messages(prompt, text, "zero_shot")
        chat_response = self.chat_completion(messages)
        return self.to_df_dict(ZERO_SHOT_TEMPLATE, messages, prompt, [[]], 0, text)

    def topic_completion(
        self, text: str, topic: str, use_chat: bool = True
    ) -> CompletionResponse:
        if not use_chat:
            prompt = "Summarize the following text into three subheadings with three corresponding bullet points. Be concise."
            prompt += "\n\nTopic:" + SEPARATOR + topic + SEPARATOR
            prompt += "\n\nText:" + SEPARATOR + text + SEPARATOR
            prompt += "\n\nSummary:"

            return self.completion(prompt)
        else:
            messages = self.create_chat_messages(topic, text, "topic")
            return self.chat_completion(messages), messages

    def topic_summarization(
        self, text: str, topic: str, use_chat: bool = True
    ) -> DfDict:
        response, messages = self.topic_completion(text, topic, use_chat)
        return self.to_df_dict(TOPIC_TEMPLATE, response, messages, [[]], 0, text)

    def in_context_completion(
        self, inputs: List[str], outputs: List[str], text: str, use_chat: bool
    ) -> CompletionResponse | ChatResponse:
        prompt = ""
        for i, o in zip(inputs, outputs):
            prompt += "Input: " + i + "\nOutput:" + o + PRIMING_SEPARATOR
        prompt += "Input:" + text + "\n" + "Output:"

        response = {}
        if not use_chat:
            response = self.completion(prompt)
        else:
            response = self.chat_completion(
                self.create_chat_messages(prompt, text, "in_context")
            )

        return response

    def in_context_prediction(
        self, examples: List[List[str]], text: str, num_examples: int, use_chat=False
    ) -> DfDict:
        res = self.in_context_completion(
            examples[0][:num_examples], examples[1][:num_examples], text, use_chat
        )
        return self.to_df_dict(IN_CONTEXT_TEMPLATE, res, examples, num_examples, text)

    def induce_completion(self, inputs: List[str], outputs: List[str], use_chat=False):
        context = "I gave a friend an instruction and some inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs.\n"
        # context = "I gave a friend an instruction and some texts. The friend read the instruction and wrote an output for every one of the texts. Here are the outputs.\n"
        if not use_chat:
            prompt = ""
            prompt_examples = ""
            for i, o in zip(inputs, outputs):
                prompt_examples += "Input:" + i + "\nOutput:" + o + PRIMING_SEPARATOR
                # prompt_examples += "Output:" + o + PRIMING_SEPARATOR

            before_pred = "The instruction was:"
            prompt += context + prompt_examples + before_pred

            return self.completion(prompt)

        else:
            messages = self.create_chat_messages(context, "", "induce")
            # for i, o in zip(inputs, outputs):
            #     messages.append(
            #         # Message("user", "Input:" + i + "\nOutput:" + o + PRIMING_SEPARATOR)
            #         Message("user", "Output:" + o + PRIMING_SEPARATOR)
            #     )

            prompt_examples = ""
            for i, o in zip(inputs, outputs):
                prompt_examples += "Input:" + i + "\nOutput:" + o + PRIMING_SEPARATOR
                # prompt_examples += "Output:" + o + PRIMING_SEPARATOR
            messages.append(Message("user", prompt_examples))

            messages.append(Message("user", "The instruction was:"))

            return self.chat_completion(messages)

    def induce_instruction(
        self, examples: List[List[str]], num_examples: int, use_chat=False
    ) -> DfDict:
        res = self.induce_completion(examples[0], examples[1], use_chat)
        return self.to_df_dict(INDUCE_TEMPLATE, res, examples, num_examples)

    def persona_completion(
        self, text: str, persona_context_setter: str, use_chat: bool = True
    ):
        strings = [
            persona_context_setter,
            "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.",
            "Text: ###\n" + text + "\n###",
            "Summary:",
        ]

        if not use_chat:
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

            return response, messages

    def generate_persona_context(topic: str):
        topics_str = topic.replace(",", ", ")
        return "You are an expert on the following topic(s): " + topics_str + "."

    def persona_summarization(
        self,
        text: str,
        topic: str,
        use_chat: bool = False,
    ):
        persona_context = self.generate_persona_context(topic)
        response, messages = self.persona_completion(text, persona_context, use_chat)
        info_dict = self.to_df_dict(
            PERSONA_TEMPLATE,
            response,
            messages,
            [[]],
            0,
            text,
        )
        return info_dict

    def improve_completion(self, text: str, use_chat: bool = True):
        prompt = "Summarize the following text into three subheadings with three corresponding bullet points. Be concise.\n"
        prompt += "Text: ###\n" + text + SEPARATOR
        prompt += "Summary:"

        if not use_chat:
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
                    "I am not satisfied with the summary. Write an improved version using three subheadings with three corresponding bullet points. Be concise.",
                )
            )
            response = self.chat_completion(messages)
            response_msg = response.choices[0].message
            messages.append(Message(response_msg.role, response_msg.content))

            return response, messages

    def improve_summarization(self, text: str, use_chat: bool = True):
        response, messages = self.improve_completion(text, use_chat)
        info_dict = self.to_df_dict(IMPROVE_TEMPLATE, response, messages, text=text)
        return info_dict

    def important_parts_completion(self, text: str, use_chat: bool = True):
        if not use_chat:
            prompt = "Find all important parts of the following text.\n"
            prompt += "Text: ###\n" + text + SEPARATOR
            prompt += "Important parts:"
            important_parts_msg = self.completion(prompt).choices[0].text

            prompt = "Here is a text and its important parts. Please improve the important parts.\n"
            prompt += "Text:" + SEPARATOR + text + SEPARATOR + "\n\n"
            prompt += "Important parts:" + SEPARATOR + important_parts_msg + SEPARATOR
            important_parts_msg = self.completion(prompt).choices[0].text

            prompt = "Here is a text and its important parts. Summarize the content into a three subheadings with three corresponding bullet points.\n"
            prompt += "Text:" + SEPARATOR + text + SEPARATOR + "\n\n"
            prompt += "Important parts:" + SEPARATOR + important_parts_msg + SEPARATOR
            response = self.completion(prompt).choices[0].text

            return response

        else:
            messages = self.create_chat_messages("", text, "important_parts")
            response = self.chat_completion(messages).choices[0].message
            messages.append(Message(response.role, response.content))
            messages.append(
                Message(
                    "user",
                    "I am not satisfied with the important parts. Please improve them.",
                )
            )
            important_parts_msg = self.chat_completion(messages).choices[0].message

            final_messages = self.create_chat_messages(
                important_parts_msg.content, text, "summarize_important_parts"
            )

            response = self.chat_completion(final_messages)
            return response, messages + final_messages

    def important_parts_summarization(self, text: str, use_chat: bool = False):
        response, messages = self.important_parts_completion(text, use_chat)
        info_dict = self.to_df_dict(
            IMPORTANT_PARTS_TEMPLATE, response, messages, text=text
        )
        return info_dict

    def template_completion(
        self,
        text: str,
        template: str,
        use_chat: bool = False,
        reference_text: str = "",
        reference_summary: str = "",
    ):
        template = "\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET"
        if not use_chat:
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

    def repeat_completion(self, text: str, use_chat: bool = False):
        messages = self.create_chat_messages("", text, "repeat")

        return self.chat_completion(messages).choices[0].message

    def repeat_summarization(self, text: str, use_chat: bool = False):
        info_dict = self.to_df_dict(
            REPEAT_TEMPLATE,
            self.repeat_completion(text, use_chat),
            text=text,
        )
        return info_dict

    def briefness_completion(self, text: str, modifier: str):
        messages = self.create_chat_messages(modifier, text, "briefness")

        return self.chat_completion(messages).choices[0].message

    def briefness_summarize(self, text: str, modifier: str):
        info_dict = self.to_df_dict(
            Template(modifier),
            self.briefness_completion(text, modifier),
            text=text,
        )
        return info_dict

    def quality_completion(self, text: str, modifier: str):
        messages = self.create_chat_messages(modifier, text, "quality")

        return self.chat_completion(messages).choices[0].message

    def quality_summarize(self, text: str, modifier: str):
        info_dict = self.to_df_dict(
            Template(modifier),
            self.quality_completion(text, modifier),
            text=text,
        )
        return info_dict
