# Always use \n###\n as separator between priming examples
from string import Template


SEPARATOR = "\n###\n"
CSV_MSG_SEPARATOR = "\n----------\n"
MAX_TOKENS_GPT3 = 4096
MAX_TOKENS_GPT4 = 8192


# TEMPLATES #
FOLLOW_UP_TEMPLATE = Template("Chat messages with follow_up_questions")
IMPROVE_TEMPLATE = Template("Chat messages, ask to improve the output")
IN_CONTEXT_TEMPLATE = Template("Input: ${text} \nOutput:")
INDUCE_TEMPLATE = Template(
    "${Context_setter} *sep* ${example_pairs} *sep* The instruction was:"
)
TOPIC_TEMPLATE = Template("${Context_setter} Topic: ${topic} Text: ${text} \nSummary:")
PERSONA_TEMPLATE = Template("${Context_setter} ${prompt} Text: ${text}\n")
