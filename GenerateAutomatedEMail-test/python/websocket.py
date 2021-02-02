import asyncio
import websockets
from model import model

def email_prediction(message):

    print(f'"{message[7:]}" wird verarbeitet')
    if()
    settings = {"checkpoint_dir": "../checkpoints", "email": "", "BATCH_SIZE": 64, "embedding_dim": 256,
                "units": 512,
                "data_size": 1000, "filepath": "../data/amazon.csv", "input": f"{message}"}
    predict = model().predict(settings)
    return predict

async def server(websocket, path):
    async for message in websocket:
        #await websocket.send(f"{message}  Nachricht angekommen")
        if message[:7] != "predict":
            predict = email_prediction(message)
            await websocket.send(f"Your Prediction is {predict}")
            print(f'"{predict}" wurde versendet')




start_server = websockets.serve(server, "127.0.0.2", 5679)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()