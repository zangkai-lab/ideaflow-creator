from pydantic import BaseModel, Field


class GetPromptModel(BaseModel):
    text: str
    need_translate: bool = Field(
        True,
        description="Whether we need to translate the input text.",
    )


class GetPromptResponse(BaseModel):
    text: str
    success: bool
    reason: str


__all__ = [
    "GetPromptResponse",
    "GetPromptModel",
]