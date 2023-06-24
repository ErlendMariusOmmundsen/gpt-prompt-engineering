# Always use \n###\n as separator between priming examples
from string import Template

BASE_PROMPT = "Summarize the text."
BEGIN_SEPARATOR = "###\n"
END_SEPARATOR = "\n###"
PRIMING_SEPARATOR = "\n###\n"
CSV_SEPARATOR = "\n\n----------\n\n"
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
TEMPLATE_TEMPLATE = Template(
    "I will provide a template. All caps is placeholder. Preserve formatting. Template: ${template}. Summarize the text: ${text}"
)
HEADINGS_FIRST_TEMPLATE = Template(
    "Summarize the text into three subheadings. Text: ${text}. Add three bullet points to each subheading."
)

# MAX LENGTHS #
BULLET_MAX_LENGTH = 60
SUBHEADING_MAX_LENGTH = 40


# MODIFIERS #
QUALITY_MODIFIERS = [
    "articulate",
    "eloquent",
    "well-written",
    "well-put",
    "well-expressed",
    "well-stated",
    "well-worded",
    "well-formulated",
    "well-presented",
    "a pleasure to read",
    "a joy to read",
    "a delight to read",
    "delightful",
    "meaningful, persuasive, and delightful",
    "masterfully written",
    "masterpieces of the English language",
    "understandable and enjoyable for everyone",
]

STRUCTURE_MODIFIERS = [
    "well-constructed",
    "well-organized",
    "well-arranged",
    "well-ordered",
    "well-structured",
    "well-phrased",
    "well-composed",
    "coherent",
    "flowing",
    "put together well",
    "well put together",
    "put together so that it flows well",
    "put together so that they are connected well",
    "put together so that there is a good flow",
]

FORMAT_MODIFIERS = [
    "3 subheadings with 3 bullet points",
    "3 subheadings, each with 3 bullet points",
    "Three subheadings, each with three bullet points",
    "3 x 3 subheadings and bullet points",
    "3 by 3 subheadings with bullet points",
    "Three subheadings with three corresponding bullet points",
    "Output must be three subheadings, each with three bullet points",
    "Output format: three subheadings, each with three bullet points",
]

# Without words that are potentially rude
LENGTH_MODIFIERS = [
    "brief",
    "short",
    "shortened",
    "abbreviated",
    "abridged",
    "curtailed",
    "trimmed",
    "less than " + str(BULLET_MAX_LENGTH) + " characters long",
]

# Words contained: marked by the use of few words to convey much information or meaning
DENSENESS_MODIFIERS = [
    "compendious",
    "comprehensive",
    "concise",
    "compressed",
    "succinct",
    "pithy",
    "terse",
    "epigrammatic",
    "telegraphic",
    "condensed",
    "crisp",
    "aphoristic",
    "compact",
    "monosyllabic",
    "laconic",
    "sententious",
    "elliptical",
    "elliptic",
    "apothegmatic",
    "significant",
    "well-turned",
]


GEVAL_FLUENCY = Template(
    """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Example:

Summary:

$summary


Evaluation Form (scores ONLY):

- Fluency (1-3):"""
)

GEVAL_RELEVANCE = Template(
    """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Example:


Source Text:

$text

Summary:

$summary


Evaluation Form (scores ONLY):

- Relevance:
    """
)


GEVAL_CONSISTENCY = Template(
    """You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Text: 

$text

Summary: 

$summary


Evaluation Form (scores ONLY):

- Consistency:
    """
)


GEVAL_COHERENCE = Template(
    """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:


Source Text:

$text

Summary:

$summary


Evaluation Form (scores ONLY):

- Coherence:
    """
)
