from dataclasses import fields
import os
import pprint
from typing import List, Tuple
import openai
import configparser
import pandas as pd
from string import Template
from constants import (
    BASE_PROMPT,
    BASE_PROMPT_TEMPLATE,
    BASELINE_TEMPLATE,
    BEST_FORMAT_MODIFIER,
    DESCRIPTION_TEMPLATE,
    END_SEPARATOR,
    GEVAL_FLUENCY,
    GEVAL_COHERENCE,
    GEVAL_CONSISTENCY,
    GEVAL_RELEVANCE,
    HEADINGS_FIRST_TEMPLATE,
    IMPORTANT_PARTS_TEMPLATE,
    IMPROVE_TEMPLATE,
    LITERARY_MODIFIERS_STRING,
    PARAPHRASE_TEMPLATE,
    PERSONA_TEMPLATE,
    REPEAT_TEMPLATE,
    BEGIN_SEPARATOR,
    PRIMING_SEPARATOR,
    MAX_TOKENS_GPT3,
    MAX_TOKENS_GPT4,
    IN_CONTEXT_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    INDUCE_TEMPLATE,
    STEP_BY_STEP_TEMPLATE,
    TEMPLATE_TEMPLATE,
    TOPIC_TEMPLATE,
    TRANSLATION_TEMPLATE,
    ZERO_SHOT_TEMPLATE,
)
from dataclss import ChatResponse, CompletionResponse, DfDict, Message
from utils import (
    messages_to_string,
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
        self.chat_model = "gpt-3.5-turbo-16k"
        self.prompt = ""
        self.suffix = ""
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
        self.temperature = 0.3
        # An alternative to sampling with temperature, called nucleus sampling, where the model considers the
        #    results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        self.top_p = 1
        # How many completions to generate for each prompt.
        self.n = 1
        # Whether to stream back partial progress.
        self.stream = False
        # Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. If logprobs is
        #    5, the API will return a list of the 5 most likely tokens.
        self.logprobs = None
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
            logprobs=self.logprobs,
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
        while True:
            try:
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
            except:
                continue

    def to_df_dict(
        self,
        prompt_template: Template,
        response,
        prompt: str = "",
        examples: List[List[str]] = [[]],
        num_examples: int = 0,
        text="",
    ) -> DfDict:
        print("\tResponse type:", type(response))
        if response.object == "text_completion":
            return DfDict(
                prompt_template.template,
                examples,
                num_examples,
                text,
                prompt=prompt,
                prediction=response.choices[0].text,
                finish_reason=response.choices[0].finish_reason,
            )
        else:
            message = response.choices[0].message
            msg_text = clean_prediction(message.content)
            return DfDict(
                prompt_template.template,
                examples,
                num_examples,
                text,
                prompt=prompt,
                prediction=msg_text,
                finish_reason=response.choices[0].finish_reason,
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
        if path == "":
            raise ValueError("Path cannot be empty")
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
            case "base_prompt":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    )
                ]

            case "in_context":
                messages = []
                # examples = prompt.split(PRIMING_SEPARATOR)
                # for example in examples:
                #     messages.append(Message("user", example))
                messages.append(Message("user", prompt + text))

            case "follow_up_questions":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
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
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "improve":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "topic":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. The text is on the following topic(s): "
                        + prompt
                        + "\nText: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "important_parts":
                messages = [
                    Message(
                        "user",
                        "Find all important parts of the following text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "summarize_important_parts":
                messages = [
                    Message(
                        "user",
                        "Here is a text and its important parts. Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\nImportant parts: "
                        + BEGIN_SEPARATOR
                        + prompt
                        + END_SEPARATOR,
                    ),
                ]

            case "induce":
                messages = [
                    Message("user", prompt + text),
                ]

            case "template":
                messages = [
                    Message(
                        "user",
                        # "I am going to provide you a template for your output. Everything in all caps is a placeholder. Please preserve the formatting and overall template that I provide. Template: "
                        "I am going to provide you a summary template. Everything in all caps is a placeholder. Please preserve the format, and number of subheadings and bullet points. Template: "
                        + prompt
                        + "\nSummarize the following text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "template_with_examples":
                messages = [
                    Message(
                        "user",
                        "I am going to provide you a template for your output. Everything in all caps is a placeholder. The prompts must result in output summaries that adhere to the template format. Template: "
                        + prompt
                        + "Here is a sample interaction after the prompt was provided: "
                        + BEGIN_SEPARATOR
                        + "\nSummarize the text. Text: "
                        + reference_text
                        + "\nChatGPT: "
                        + reference_summary
                        + END_SEPARATOR,
                    ),
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    ),
                ]

            case "headings_first":
                messages = [
                    Message(
                        "user",
                        "Summarize the following text into three subheadings. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    )
                ]

            case "repeat":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. "
                        + LITERARY_MODIFIERS_STRING
                        + BEST_FORMAT_MODIFIER
                        + ". Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\nSummarize the text.",
                    ),
                ]

            case "description":
                messages = [
                    Message(
                        "user",
                        "Here is a text and its description. Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\nDescription: "
                        + BEGIN_SEPARATOR
                        + prompt
                        + END_SEPARATOR,
                    )
                ]

            case "step_by_step":
                messages = [
                    Message(
                        "user",
                        "Identify and analyze the text's structure and meaning. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "Let's think step by step.",
                    )
                ]

            case "analyze_main_themes":
                messages = [
                    Message(
                        "user",
                        "Analyze the text and identify three main themes or topics discussed in the text. For each theme, provide three supporting points from the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    )
                ]

            case "main_points":
                messages = [
                    Message(
                        "user",
                        "Write an output summarizing the main points of the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR,
                    )
                ]

            case "format":
                messages = [
                    Message(
                        "user",
                        BASE_PROMPT
                        + "\nText: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\n"
                        + prompt
                        + ".",
                    ),
                ]

            case "length" | "quality" | "structure" | "denseness" | "relevance":
                messages = [
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\nThe bullet points and subheadings must be "
                        + prompt
                        + ".",
                    ),
                ]

            case "shorten_as_possible":
                shorten_modifier = "Shorten the answer as possible. "
                messages = [
                    Message(
                        "user",
                        "Summarize the text. Text: "
                        + BEGIN_SEPARATOR
                        + text
                        + END_SEPARATOR
                        + "\n"
                        + shorten_modifier,
                    )
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
                examples = prompt.split(BEGIN_SEPARATOR)
                for example in examples:
                    messages.append(
                        Message(BEGIN_SEPARATOR + example + BEGIN_SEPARATOR, "user")
                    )
                messages.append(
                    Message(
                        "assistant",
                        "Before summarizing, I will ask two questions about the text.",
                    )
                )

        if not approach in [
            # "template",
            # "template_with_examples",
            "follow_up_questions",
            "important_parts",
            "headings_first",
            "format",
            "step_by_step",
            "induce",
        ]:
            messages[-1].content += "\n" + LITERARY_MODIFIERS_STRING
            messages[-1].content += "\n" + BEST_FORMAT_MODIFIER + "."

        return messages

    def baseline_summarization(self, text: str) -> DfDict:
        current_strategy = "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n"
        messages = [Message("user", current_strategy + text)]
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return self.to_df_dict(
            BASELINE_TEMPLATE, response, messages_to_string(messages), [[]], 0, text
        )

    def base_prompt_summarization(self, text: str) -> DfDict:
        messages = self.create_chat_messages("", text, "base_prompt")
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return self.to_df_dict(
            BASE_PROMPT_TEMPLATE, response, messages_to_string(messages), [[]], 0, text
        )

    def paraphrase(self, text: str) -> DfDict:
        prompt = PARAPHRASE_TEMPLATE.substitute(text=text)
        messages = [Message("user", prompt)]
        response = self.chat_completion(messages)

        return self.to_df_dict(PARAPHRASE_TEMPLATE, response, prompt, [[]], 0, text)

    def translate(self, text: str, to_lang: str = "German") -> Tuple[ChatResponse, str]:
        prefix = "Translate the following text into " + to_lang + ": "
        prompt = prefix + text
        prompt += "\nTranslation:"
        messages = [Message("user", prompt)]
        response = self.chat_completion(messages)

        return response, prompt

    def translation_paraphrase(self, text: str, to_lang: str = "German") -> DfDict:
        response, _ = self.translate(text, to_lang)
        response2, prompt2 = self.translate(
            response.choices[0].message.content, "English"
        )

        return self.to_df_dict(TRANSLATION_TEMPLATE, response2, prompt2, [[]], 0, text)

    def follow_up_completion(self, text: str) -> Tuple[ChatResponse, str]:
        messages = self.create_chat_messages("", text, "follow_up_questions")
        role = "assistant"
        for _ in range(2):
            response = self.chat_completion(messages)
            message = response.choices[0].message
            messages.append(Message(role, message.content))
            role = "user" if role == "assistant" else "assistant"

        final = (
            "Now summarize the text. "
            + LITERARY_MODIFIERS_STRING
            + " "
            + BEST_FORMAT_MODIFIER
            + "."
        )
        messages.append(Message("user", final))

        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return response, messages_to_string(messages)

    def follow_up_summarization(self, text) -> DfDict:
        response, messages = self.follow_up_completion(text)
        return self.to_df_dict(FOLLOW_UP_TEMPLATE, response, messages, [[]], 0, text)

    def zero_shot_summarization(self, text: str, approach: str) -> DfDict:
        messages = self.create_chat_messages("", text=text, approach=approach)
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        return self.to_df_dict(
            ZERO_SHOT_TEMPLATE, response, messages_to_string(messages), [[]], 0, text
        )

    def topic_completion(
        self, text: str, topic: str, use_chat: bool = True
    ) -> Tuple[CompletionResponse | ChatResponse, str]:
        if not use_chat:
            prompt = "Summarize the following text into three subheadings with three corresponding bullet points."
            prompt += "\n\nTopic: " + BEGIN_SEPARATOR + topic + END_SEPARATOR
            prompt += "\n\nText: " + BEGIN_SEPARATOR + text + END_SEPARATOR
            prompt += "\n\nSummary:"

            return self.completion(prompt), prompt
        else:
            messages = self.create_chat_messages(topic, text, "topic")
            response = self.chat_completion(messages)
            messages.append(Message("assistant", response.choices[0].message.content))
            return response, messages_to_string(messages)

    def topic_summarization(
        self, text: str, topic: str, use_chat: bool = True
    ) -> DfDict:
        topic_str = topic.replace(",", ", ") + "."
        response, prompt = self.topic_completion(text, topic_str, use_chat)
        return self.to_df_dict(TOPIC_TEMPLATE, response, prompt, [[]], 0, text)

    def in_context_completion(
        self,
        inputs: List[str],
        outputs: List[str],
        text: str,
        only_outputs: bool,
        use_chat: bool,
    ) -> Tuple[CompletionResponse | ChatResponse, str]:
        prompt = (
            "Summarize the text. "
            + BEST_FORMAT_MODIFIER
            + ". "
            + LITERARY_MODIFIERS_STRING
            + "\nFollowing are some good summary examples.\n"
            + PRIMING_SEPARATOR
        )
        for i, o in zip(inputs, outputs):
            if only_outputs:
                prompt += "Text: [Text]" + "\nSummary: " + o + PRIMING_SEPARATOR
            else:
                prompt += "Text: " + i + "\nSummary: " + o + PRIMING_SEPARATOR
        prompt += "Text: " + text + "\n" + "Summary:"

        response = {}
        if not use_chat:
            response = self.completion(prompt)
        else:
            messages = [Message("user", prompt)]
            response = self.chat_completion(messages)
            messages.append(Message("assistant", response.choices[0].message.content))
            prompt = messages_to_string(messages)

        return response, prompt

    def in_context_summarization(
        self,
        text: str,
        examples: List[List[str]],
        num_examples: int,
        only_outputs=False,
        use_chat=True,
    ) -> DfDict:
        res, prompt = self.in_context_completion(
            examples[0][:num_examples],
            examples[1][:num_examples],
            text,
            only_outputs,
            use_chat,
        )
        return self.to_df_dict(
            prompt_template=IN_CONTEXT_TEMPLATE,
            response=res,
            prompt=prompt,
            examples=examples,
            num_examples=num_examples,
            text=text,
        )

    def induce_completion(
        self,
        inputs: List[str],
        outputs: List[str],
        only_outputs=False,
        use_chat=False,
    ) -> Tuple[ChatResponse | CompletionResponse, str]:
        context = (
            "I gave a friend an instruction and some inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs.\n"
            if not only_outputs
            else "I gave a friend an instruction and some texts. The friend read the instruction and wrote an output for every one of the texts. Here are the outputs.\n"
        )
        prompt_examples = PRIMING_SEPARATOR
        for i, o in zip(inputs, outputs):
            prompt_examples += (
                "Input: " + i + "\nOutput: " + o + PRIMING_SEPARATOR
                if not only_outputs
                else "Input: [Text]\nOutput: " + o + PRIMING_SEPARATOR
            )

        if not use_chat:
            before_pred = "The instruction was:"
            prompt = context + prompt_examples + before_pred
            return self.completion(prompt), prompt

        else:
            messages = self.create_chat_messages(context, prompt_examples, "induce")
            messages.append(Message("user", "The instruction was:"))
            response = self.chat_completion(messages)
            messages.append(Message("assistant", response.choices[0].message.content))
            return response, messages_to_string(messages)

    def induce_instruction(
        self,
        examples: List[List[str]],
        num_examples: int,
        only_outputs: bool = False,
        use_chat: bool = False,
    ) -> DfDict:
        res, prompt = self.induce_completion(
            examples[0][:num_examples],
            examples[1][:num_examples],
            only_outputs,
            use_chat,
        )
        return self.to_df_dict(INDUCE_TEMPLATE, res, prompt, examples, num_examples)

    def persona_completion(
        self, text: str, persona_context_setter: str, use_chat: bool = True
    ):
        strings = [
            persona_context_setter,
            "Summarize the text.",
            "Text: " + BEGIN_SEPARATOR + text + END_SEPARATOR,
            "Summary:",
        ]

        if not use_chat:
            prompt = ""
            for s in strings:
                prompt += s

            return self.completion(prompt), prompt

        else:
            messages = self.create_chat_messages(
                persona_context_setter, text, "persona"
            )
            response = self.chat_completion(messages)
            messages.append(
                Message(
                    response.choices[0].message.role,
                    response.choices[0].message.content,
                )
            )

            return response, messages_to_string(messages)

    def generate_persona_context(self, topic: str):
        topics_str = topic.replace(",", ", ")
        return "You are an expert on the following topic(s): " + topics_str + "."

    def persona_summarization(
        self,
        text: str,
        topic: str,
        use_chat: bool = False,
    ) -> DfDict:
        persona_context = self.generate_persona_context(topic)
        response, prompt = self.persona_completion(text, persona_context, use_chat)
        info_dict = self.to_df_dict(
            PERSONA_TEMPLATE,
            response,
            prompt,
            [[]],
            0,
            text,
        )
        return info_dict

    def improve_completion(
        self, text: str, use_chat: bool = True
    ) -> Tuple[CompletionResponse | ChatResponse, str]:
        prompt = "Summarize the following text into three subheadings with three corresponding bullet points.\n"
        prompt += "Text: " + BEGIN_SEPARATOR + text + END_SEPARATOR
        prompt += "\nSummary:"

        if not use_chat:
            response = self.completion(prompt)
            summary = response.choices[0].text
            prompt = "Here is a text and its summary. Please improve the summary."
            prompt += "\nText: " + BEGIN_SEPARATOR + text + END_SEPARATOR
            prompt += "\nSummary: " + BEGIN_SEPARATOR + summary + END_SEPARATOR
            return self.completion(prompt), prompt

        else:
            messages = self.create_chat_messages(prompt, text, "improve")
            response_msg = self.chat_completion(messages).choices[0].message
            messages.append(Message(response_msg.role, response_msg.content))
            messages.append(
                Message(
                    "user",
                    "I am not satisfied with the summary. Write an improved version. "
                    + LITERARY_MODIFIERS_STRING
                    + " "
                    + BEST_FORMAT_MODIFIER
                    + ".",
                )
            )
            response = self.chat_completion(messages)
            response_msg = response.choices[0].message
            messages.append(Message(response_msg.role, response_msg.content))

            return response, messages_to_string(messages)

    def improve_summarization(self, text: str, use_chat: bool = True) -> DfDict:
        response, prompt = self.improve_completion(text, use_chat)
        info_dict = self.to_df_dict(IMPROVE_TEMPLATE, response, prompt, text=text)
        return info_dict

    def important_parts_completion(
        self, text: str, use_chat: bool = True
    ) -> Tuple[CompletionResponse | ChatResponse, str]:
        if not use_chat:
            prompt = "Find all important parts of the following text.\n"
            prompt += "Text: " + BEGIN_SEPARATOR + text + END_SEPARATOR

            important_parts_msg = self.completion(prompt).choices[0].text
            prompt = "Here is a text and its important parts. Please improve the important parts.\n"
            prompt += "Text: " + BEGIN_SEPARATOR + text + END_SEPARATOR
            prompt += (
                "\nImportant parts:"
                + BEGIN_SEPARATOR
                + important_parts_msg
                + END_SEPARATOR
            )
            important_parts_msg = self.completion(prompt).choices[0].text

            prompt = "Here is a text and its important parts. Summarize the content into a three subheadings with three corresponding bullet points."
            prompt += "\nText: " + BEGIN_SEPARATOR + text + END_SEPARATOR
            prompt += (
                "Important parts: "
                + BEGIN_SEPARATOR
                + important_parts_msg
                + END_SEPARATOR
            )
            response = self.completion(prompt)

            return response, prompt

        else:
            messages = self.create_chat_messages("", text, "important_parts")
            response = self.chat_completion(messages).choices[0].message
            messages.append(Message(response.role, response.content))
            # messages.append(
            #     Message(
            #         "user",
            #         "I am not satisfied with the important parts. Please improve them.",
            #     )
            # )
            # important_parts_msg = self.chat_completion(messages).choices[0].message

            final_messages = self.create_chat_messages(
                response.content, text, "summarize_important_parts"
            )

            response = self.chat_completion(final_messages)
            final_messages.append(
                Message("assistant", response.choices[0].message.content)
            )
            return response, messages_to_string(messages + final_messages)

    def important_parts_summarization(self, text: str, use_chat: bool = True) -> DfDict:
        response, prompt = self.important_parts_completion(text, use_chat)
        info_dict = self.to_df_dict(
            IMPORTANT_PARTS_TEMPLATE, response, prompt, text=text
        )
        return info_dict

    def step_by_step_summarization(self, text: str, use_chat: bool = True) -> DfDict:
        messages = self.create_chat_messages("", text, "step_by_step")
        response = self.chat_completion(messages)
        messages.append(
            Message(
                response.choices[0].message.role, response.choices[0].message.content
            )
        )

        messages.append(
            Message(
                "user",
                "Summarize the text. "
                + LITERARY_MODIFIERS_STRING
                + " "
                + BEST_FORMAT_MODIFIER
                + ".",
            )
        )
        final_response = self.chat_completion(messages)
        messages.append(
            Message(
                final_response.choices[0].message.role,
                final_response.choices[0].message.content,
            )
        )

        return self.to_df_dict(
            STEP_BY_STEP_TEMPLATE,
            final_response,
            messages_to_string(messages),
            text=text,
        )

    def description_summarization(
        self, text: str, description: str, use_chat: bool = True
    ) -> DfDict:
        messages = self.create_chat_messages(description, text, "description")
        response = self.chat_completion(messages)
        messages.append(
            Message(
                response.choices[0].message.role, response.choices[0].message.content
            )
        )
        return self.to_df_dict(
            DESCRIPTION_TEMPLATE, response, messages_to_string(messages), text=text
        )

    def template_completion(
        self,
        text: str,
        reference_text: str = "",
        reference_summary: str = "",
        use_chat: bool = True,
    ) -> Tuple[CompletionResponse | ChatResponse, str]:
        template = "SUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET\nSUBHEADING\n- BULLET\n- BULLET\n- BULLET"
        if not use_chat:
            prompt = "I am going to provide you a template for your output. Everything in all caps is a placeholder. The prompts must result in output summaries that follow the template format. Template: "
            prompt += template
            prompt += (
                "\nA sample interaction after the prompt was provided is shown below.\n"
            )
            prompt += (
                BEGIN_SEPARATOR
                + "User: Summarize the text into three subheadings with three corresponding bullet points."
            )
            prompt += "\nText: " + reference_text
            prompt += "\nChatGPT: " + reference_summary + END_SEPARATOR
            prompt += "\nSummarize text. Text: " + text
            return self.completion(prompt), prompt

        else:
            if reference_text:
                messages = self.create_chat_messages(
                    template,
                    text,
                    "template_with_examples",
                    reference_text,
                    reference_summary,
                )
                response = self.chat_completion(messages)
                messages.append(
                    Message("assistant", response.choices[0].message.content)
                )
                return response, messages_to_string(messages)
            else:
                messages = self.create_chat_messages(template, text, "template")
                response = self.chat_completion(messages)
                messages.append(
                    Message("assistant", response.choices[0].message.content)
                )
                return response, messages_to_string(messages)

    def template_summarization(
        self,
        text: str,
        reference_text: str = "",
        reference_summary: str = "",
        use_chat: bool = True,
    ) -> DfDict:
        # response, prompt = self.template_completion(
        #     text, reference_text, reference_summary, use_chat
        # )
        response, prompt = self.template_completion(text, "", "", use_chat)
        info_dict = self.to_df_dict(TEMPLATE_TEMPLATE, response, prompt, text=text)
        return info_dict

    def repeat_summarization(self, text: str, use_chat: bool = True) -> DfDict:
        messages = self.create_chat_messages("", text, "repeat")
        response = self.chat_completion(messages)
        messages.append(Message("assistant", response.choices[0].message.content))
        prompt = messages_to_string(messages)

        info_dict = self.to_df_dict(
            REPEAT_TEMPLATE,
            response,
            prompt,
            text=text,
        )
        return info_dict

    def headings_first_summarization(self, text: str, use_chat: bool = True) -> DfDict:
        messages = self.create_chat_messages("", text, "headings_first")
        response_content = self.chat_completion(messages).choices[0].message
        messages.append(Message(response_content.role, response_content.content))

        messages.append(
            Message(
                "user",
                "Add three bullet points to each subheading. "
                + LITERARY_MODIFIERS_STRING,
            )
        )
        response = self.chat_completion(messages)
        messages.append(
            Message(
                response.choices[0].message.role, response.choices[0].message.content
            )
        )
        prompt = messages_to_string(messages)

        info_dict = self.to_df_dict(
            HEADINGS_FIRST_TEMPLATE,
            response,
            prompt,
            text=text,
        )
        return info_dict

    def modifier_summarize(self, text: str, modifier: str, modifies: str) -> DfDict:
        messages = self.create_chat_messages(modifier, text, modifies)
        response = self.chat_completion(messages)
        messages.append(
            Message(
                response.choices[0].message.role, response.choices[0].message.content
            )
        )
        prompt = messages_to_string(messages)

        info_dict = self.to_df_dict(
            Template(modifier),
            response,
            prompt,
            text=text,
        )
        return info_dict

    def geval(self, text: str, summary: str, metric: str):
        previous_temperature = self.temperature
        previous_logprobs = self.logprobs
        previous_n = self.n

        self.logprobs = 5
        self.n = 20
        self.temperature = 1.0
        messages = []

        match metric:
            case "fluency":
                prompt = GEVAL_FLUENCY.substitute(text=text, summary=summary)
                messages.append(Message("user", prompt))
            case "coherence":
                prompt = GEVAL_CONSISTENCY.substitute(text=text, summary=summary)
                messages.append(Message("user", prompt))
            case "consistency":
                prompt = GEVAL_COHERENCE.substitute(text=text, summary=summary)
                messages.append(Message("user", prompt))
            case "relevance":
                prompt = GEVAL_RELEVANCE.substitute(text=text, summary=summary)
                messages.append(Message("user", prompt))

        response = self.chat_completion(messages)
        prob_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        geval_score = 0

        for choice in response.choices:
            msg = choice.message.content
            score = 0
            try:
                first_word = msg.split()[0]
                score = int(float(first_word))
            except:
                try:
                    last_word = msg.split()[-1]
                    score = int(float(last_word))
                except:
                    self.n -= 1
                    continue

            score = 5 if score > 5 else score
            score = 1 if score < 1 else score
            prob_dist[score] += 1

        for score in prob_dist.keys():
            occurrences = prob_dist[score]
            score_probability = occurrences / self.n
            geval_score += score * score_probability

        self.logprobs = previous_logprobs
        self.n = previous_n
        self.temperature = previous_temperature

        return round(geval_score, 2)

    def list_all_models(self):
        model_list = openai.Model.list()["data"]
        model_ids = [x["id"] for x in model_list]
        model_ids.sort()
        pprint.pprint(model_ids)
        if "gpt-4-32k" in model_ids:
            print("##################################################")
            print("##################################################")
            print("##################################################")
            print("\ngpt-4-32k IS PUBLISHED\n")
            print("##################################################")
            print("##################################################")
            print("##################################################")
        else:
            print("\n\ngpt-4-32k is not out yet :(")
