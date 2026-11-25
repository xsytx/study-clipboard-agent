"""Validated structured output for study flashcards."""

from typing import List

from pydantic import BaseModel, Field


class Flashcard(BaseModel):
    """One question/answer pair with a short topic tag."""

    front: str = Field(min_length=1, description="A clear question or study prompt")
    back: str = Field(min_length=1, description="A concise, accurate answer")
    tag: str = Field(min_length=1, description="One short topic tag, without spaces")


class FlashcardSet(BaseModel):
    """The schema the LLM must return when generating flashcards."""

    cards: List[Flashcard] = Field(min_length=1, max_length=12)
