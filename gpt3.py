import openai
import configparser

config = configparser.ConfigParser()
config.read(".env")

openai.api_key = config["keys"]["OPENAI_API_KEY"]
openai.organization = config["keys"]["OPENAI_ORG_KEY"]


def current_summarize(text, model="text-davinci-003"):
    current_strategy = "suggest three insightful concise subheadings which summarize this text, suggest threee bullet points for each subheading:\n"
    response = openai.Completion.create(
        model=model,
        prompt=current_strategy + text,
        temperature=0,
        max_tokens=200,
    )
    return response


def summarize(text, prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt="Text: " + steve_jobs_speech + "\nBullet points:",
        temperature=0,
        max_tokens=200,
    )
    return response
