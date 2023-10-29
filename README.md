# GPT prompt engineering for abstractive summarization
Python project for running text summarization prompt experiments using OpenAI models.
The summary quality of the experiment results are saved to CSV files. The base
classes GPT in `gpt.py` and Evaluator in `evaluator.py` can be used to create new pipelines with new prompts. 

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
- Evaluation scores for:
  - BERTScore
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L
  - Average Error Count Score
  - Errors
  - Contradiction Ratio
  - Neutral Contradiction Ratio
  - Number Hallucinations
  - Three-by-three format adherence
  - Long Subheadings
  - Long Bullets
  - G-EVAL fluency
  - G-EVAL relevance
  - G-EVAL coherence
  - G-EVAL consistency
- API Config


## How to run
Prerequisites:
- A .ENV file with the following contents:
  - ```
    [keys]
    OPENAI_API_KEY=*key*
    OPENAI_ORG_KEY=*key*
    HUGGINGFACE_KEY=*key*
- Python 3.10 or newer with the required packages in `./requirements.txt` installed
- A GPU for evaluation with bert_score enabled
