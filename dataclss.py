from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatChoice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class CompletionChoice:
    text: str
    index: int
    logprobs: object
    finish_reason: str


@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    id: str
    object: str
    created: int
    choices: List[ChatChoice]
    usage: Usage


@dataclass
class CompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


@dataclass
class DfDict:
    prompt_template: str = ""
    examples: List[List[str]] = field(default_factory=list)
    num_examples: int = 0
    text: str = ""
    title: str = ""
    prompt: str = ""
    prediction: str = ""
    finish_reason: str = ""
    bert_score: List[float] = field(default_factory=list)
    rouge_1: List[float] = field(default_factory=list)
    rouge_2: List[float] = field(default_factory=list)
    rouge_L: List[float] = field(default_factory=list)
    slor: float = 0.0
    avg_error_count_score: float = 0.0
    errors: int = 0
    contradiction_ratio: float = 0.0
    neutral_contradiction_ratio: float = 0.0
    number_hallucinations: int = 0
    three_by_three: int = 0
    long_subheadings: int = 0
    long_bullets: int = 0
    geval_fluency: float = 0.0
    geval_relevance: float = 0.0
    geval_coherence: float = 0.0
    geval_consistency: float = 0.0
