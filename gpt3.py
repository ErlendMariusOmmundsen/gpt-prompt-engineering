import openai
import configparser
import pandas as pd
from string import Template

# Always use \n###\n as seperator between priming examples
seperator = "\n###\n"


class gpt3:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(".env")
        openai.api_key = config["keys"]["OPENAI_API_KEY"]
        openai.organization = config["keys"]["OPENAI_ORG_KEY"]

        self.model = "text-davinci-003"
        self.prompt = ""
        self.suffix = ""
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
        self.temperature = 1
        # An alternative to sampling with temperature, called nucleus sampling, where the model considers the
        #    results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        self.top_p = 1
        self.max_tokens = 200
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

    def completion(self, prompt):
        print(self)
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
        )

    def current_summarize(self, text):
        current_strategy = "suggest three insightful concise subheadings which summarize this text, suggest threee bullet points for each subheading:\n"
        response = self.completion(current_strategy + text)
        return response

    def simple_sum(self, text):
        # Bullet points
        bullet_response = self.completion("Text: " + text + "\nBullet points:")
        # Heading
        heading_response = self.completion(
            "Text: " + text + "\nHeading:",
        )

        return heading_response, bullet_response

    def in_context_prediction(self, inputs, outputs, text):
        prompt_examples = ""
        for i, o in zip(inputs, outputs):
            prompt_examples += "Input: " + i + "\nOutput: " + o + seperator
        prompt_examples += "Input: " + text + "\nOutput:"

        temp = Template("Input: ${text} \nOutput:")
        prompt = prompt_examples + "Input: " + text + "\nOutput:"

        response = self.completion(prompt)
        return temp.template, response

    # Params: examples = [[input, input, ...], [output, output, ...]], text = to be summarized
    def in_context_predictions(self, examples, text, num_examples):
        # TODO: Select random num of examples?
        prompt_template, res = self.in_context_prediction(
            examples[0][:num_examples], examples[1][:num_examples], text
        )
        return self.to_info_dict(prompt_template, res, examples, num_examples, text)

    # TODO: Add evaluation scores
    def to_info_dict(self, prompt_template, response, examples, num_examples, text=""):
        obj = {}
        obj["prompt_template"] = prompt_template
        obj["examples"] = examples
        obj["num_examples"] = num_examples
        obj["text"] = text
        obj["prediction"] = response.choices[0].text
        obj["finish_reason"] = response.choices[0].finish_reason
        return obj

    def save_df(self, info_dict, path):
        conf = self.get_config()
        data = []
        df = pd.DataFrame(
            [[info_dict[key] for key in info_dict.keys()] + [conf]],
            columns=[*list(info_dict.keys()), "config"],
        )
        df.to_csv(path, mode="a", index=False, header=False)

    def in_context_pipe(self, examples, text, num_examples):
        info_dict = self.in_context_predictions(examples, text, num_examples)
        self.save_df(info_dict, "in-context.csv")

    def induce_instruction(self, inputs, outputs, num_examples):
        prompt = ""
        context = (
            "I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:"
            + seperator
        )
        prompt_examples = ""
        for i, o in zip(inputs, outputs):
            prompt_examples += "Input: " + i + "\nOutput: " + o + seperator
        before_pred = "The instruction was"

        prompt += context + prompt_examples + before_pred
        temp = Template("Context_setter *sep* example_pairs *sep* The instruction was")

        return temp.template, self.completion(prompt)

    def induce_instructions(self, examples, num_examples):
        # TODO: Select random num of examples?
        prompt_template, res = self.induce_instruction(
            examples[0][:num_examples], examples[1][:num_examples], 2
        )
        return self.to_info_dict(prompt_template, res, examples, num_examples)

    def induce_pipe(self, examples, num_examples):
        info_dict = self.induce_instructions(examples, num_examples)
        self.save_df(info_dict, "instruction-induction.csv")
