@dataclass
class Message:
    role: str
    content: str

    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__)


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
    prompt_template: str
    examples: List[str]
    num_examples: int
    text: str
    prediction: str
    finish_reason: str
