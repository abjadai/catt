from fastapi import FastAPI

from api.catt_service import tashkeel_text
from api.models import TashkeelRequest, TashkeelResponse

app = FastAPI()


@app.post("/tashkeel", response_model=TashkeelResponse)
async def tashkeel(request: TashkeelRequest):
    diacritized_text = tashkeel_text(request.text)
    return TashkeelResponse(
        original_text=request.text, diacritized_text=diacritized_text
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
