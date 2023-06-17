# Always use \n###\n as separator between priming examples
from string import Template


SEPARATOR = "\n###"
PRIMING_SEPARATOR = "\n###\n"
CSV_MSG_SEPARATOR = "\n----------\n"
MAX_TOKENS_GPT3 = 4096
MAX_TOKENS_GPT4 = 8192
# MAX_TOKENS_GPT4 = 32768 When the GPT-4-32K model is released


# TEMPLATES #
BASELINE_TEMPLATE = Template(
    "suggest three insightful, concise subheadings which summarize this text, suggest three bullet points for each subheading:\n ${text}"
)
FOLLOW_UP_TEMPLATE = Template("Chat messages with follow_up_questions")
IMPROVE_TEMPLATE = Template(
    "Summarize the text[...] Text: ${text}. Here is a text and its summary. Please improve the summary.\n Text: ${text} \n Summary: ${summary}"
)
IMPORTANT_PARTS_TEMPLATE = Template(
    "Find important parts of the text. Text: ${text}. Improve them.\n Summarize the text. Important parts: ${important_parts} Text: ${text}"
)
IN_CONTEXT_TEMPLATE = Template("Input: ${text} \nOutput:")
INDUCE_TEMPLATE = Template(
    "${Context_setter} *sep* ${example_pairs} *sep* The instruction was:"
)
TOPIC_TEMPLATE = Template(
    "${Context_setter} The text is on the following topic(s): ${topic} Text: ${text}"
)
PERSONA_TEMPLATE = Template("${Context_setter} ${prompt} Text: ${text}\n")
REPEAT_TEMPLATE = Template("${Prompt} Text: ${text} ${Prompt}")
ZERO_SHOT_TEMPLATE = Template("${Prompt} Text: ${text} ")

# MAX LENGTHS #
BULLET_MAX_LENGTH = 60
SUBHEADING_MAX_LENGTH = 40
