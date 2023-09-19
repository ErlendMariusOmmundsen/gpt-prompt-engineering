# GPT prompt engineering for abstractive summarization
Repo for testing prompts for abstractive text summarization on OpenAI models and evaluating output summary quality. The base
classes GPT in `gpt.py` and Evaluator in `evaluator.py` can be used to create new pipelines for new prompts. 

## TED Talk Transcript Summarization Dataset
The dataset CSV file is called `manual_summaries2.csv` and can be found in `/data`. 

## Results 
All results from experiments are stored in CSV files that contain columns for:
- Template used
- In-context learning examples
- Number of examples
- Input text
- Input text title
- Prompt
- Prediction
- Finish reason
- Evaluation scores for bert_score, rouge_1, rouge_2, rouge_L, slor, avg_error_count_score, errors, contradiction_ratio, neutral_contradiction_ratio, number_hallucinations, three_by_three,long_subheadings, long_bullets, geval_fluency, geval_relevance, geval_coherence, geval_consistency
- API Config


## How to run
All scripts and notebooks used to generate and inspect results are in `/scripts` or `./`. The scripts were edited when experimenting.

Prerequisites:
- A .ENV file with the following contents:
  - ```
    [keys]
    OPENAI_API_KEY=*key*
    OPENAI_ORG_KEY=*key*
    HUGGINGFACE_KEY=*key*
- Python 3.10 or newer
- A GPU for evaluation with bert_score enabled
