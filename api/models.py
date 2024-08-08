from pydantic import BaseModel


class TashkeelRequest(BaseModel):
    text: str


class TashkeelResponse(BaseModel):
    original_text: str
    diacritized_text: str
