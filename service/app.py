import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse


async def load_predictor():
    from src.inference.infer import Predictor
    global predictor
    predictor = Predictor(r'C:\Users\Sololearn\Desktop\london_house_price_prediction\src\train\final_model.pt')


async def predict(request):
    payload = await request.json()
    response = predictor.infer(payload)
    return JSONResponse(response)

on_startup = [load_predictor]
routes = [Route('/predict', predict, methods=['POST'])]
app = Starlette(routes=routes, on_startup=on_startup)

if __name__ == "__main__":
    uvicorn.run(
        'app:app',
        host='0.0.0.0',
        port=8000,
        reload=False
    )

