from typing import Literal

from pydantic import BaseModel, Field


class TextDocument(BaseModel):
    title: str = Field(description="Document title; used as the citation shown with search results", examples=["Refund policy"])
    text: str = Field(description="Full document text")


class SearchRequest(BaseModel):
    query: str = Field(description="The question, in everyday language", examples=["What are my fundamental rights?"])
    limit: int = Field(10, description="Maximum number of results")
    types: list[str] | None = Field(None, description="Only these point types: chunk, qa_pair, topic, summary. Omit for all")
    topics: list[str] | None = Field(None, description="Only points tagged with these topics. Omit for all")
    graph: bool | None = Field(None, description="Also use chunks tied to the question's entities in the knowledge graph. Omit to follow `GRAPH_SEARCH`")


class AskRequest(BaseModel):
    query: str = Field(description="The question, in everyday language", examples=["What are my fundamental rights?"])
    types: list[str] | None = Field(None, description="Only use these point types as context. Omit for all")
    topics: list[str] | None = Field(None, description="Only use points tagged with these topics. Omit for all")
    graph: bool | None = Field(None, description="Also use chunks tied to the question's entities in the knowledge graph. Omit to follow `GRAPH_SEARCH`")


class EvalQuestion(BaseModel):
    id: str = Field(description="Unique id of the question", examples=["q-article-21"])
    question: str = Field(description="The question, in everyday language", examples=["Can the government take away my life or liberty?"])
    evidence: list[str] = Field(min_length=1, description="Phrases copied from the text that answers it; a section containing any of them is relevant", examples=[["procedure established by law"]])
    split: Literal["train", "test"] = Field("test", description="Questions in `train` may be used to fine-tune; only `test` is reported by default. Split by Part so fine-tuning cannot leak")
    note: str = Field("", description="Free text, e.g. who wrote the question")


class EvalQuestions(BaseModel):
    questions: list[EvalQuestion]


class EvalRun(BaseModel):
    retrievers: list[str] | None = Field(None, description="bm25, base, tuned. Omit for all available", examples=[["bm25", "base"]])
    k: int = Field(5, ge=1, le=20, description="Cut-off for Recall@k and nDCG@k")
    split: Literal["test", "train", "all"] = Field("test", description="Which questions to score")
