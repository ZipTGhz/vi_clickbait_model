from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import PredictRequest, PredictResponse

app = FastAPI(title='ViClickbait Detection API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/api/v1/detect', response_model=PredictResponse)
async def detect(payload: PredictRequest):
    # Placeholder: Kết nối hàm predict_clickbait từ src/inference.py tại đây
    return {
        'label': 'non-clickbait',
        'confidence': 0.95,
        'is_clickbait': False
    }
