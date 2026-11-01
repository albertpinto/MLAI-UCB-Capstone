from pydantic import BaseModel, Field


class TextDocument(BaseModel):
    title: str = Field(description="Document title; used as the citation shown with search results", examples=["Refund policy"])
    text: str = Field(description="Full document text")


class SearchRequest(BaseModel):
    query: str = Field(description="The question, in everyday language", examples=["What are my fundamental rights?"])
    limit: int = Field(10, description="Maximum number of results")
    types: list[str] | None = Field(None, description="Only these point types: chunk, qa_pair, topic, summary. Omit for all")
    topics: list[str] | None = Field(None, description="Only points tagged with these topics. Omit for all")


class AskRequest(BaseModel):
    query: str = Field(description="The question, in everyday language", examples=["What are my fundamental rights?"])
    types: list[str] | None = Field(None, description="Only use these point types as context. Omit for all")
    topics: list[str] | None = Field(None, description="Only use points tagged with these topics. Omit for all")
