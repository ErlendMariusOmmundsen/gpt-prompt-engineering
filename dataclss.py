from dataclasses import dataclass
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
    examples: List[List[str]] = None
    num_examples: int = 0
    text: str = ""
    prompt: str = ""
    prediction: str = ""
    finish_reason: str = ""
    bert_score: float = 0.0
    rogue_1: float = 0.0
    rogue_2: float = 0.0
    rogue_L: float = 0.0
    slor: float = 0.0
    avg_error_count_score: float = 0.0
    errors: int = 0
    contradiction_ratio: float = 0.0
    neutral_contradiction_ratio: float = 0.0
    number_hallucinations: int = 0
    three_by_three: int = 0
    long_subheadings: int = 0
    long_bullets: int = 0
