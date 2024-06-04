from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from datetime import datetime
from pydantic import BaseModel


app = FastAPI()

# Montar el directorio de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar el modelo Random Forest
rf_model = joblib.load('models/rf_model.pkl.gz')

# Cargar el modelo SARIMAX (anteriormente ARIMA)
model_sarimax = joblib.load('models/sarimax_model.pkl.gz')

# Configurar las plantillas
templates = Jinja2Templates(directory="templates")

class RandomForestInput(BaseModel):
    fecha: str
    hora: int
    codigo_localidad: int
    edad: int
    genero: int
    tipo_incidente: int

class SarimaxInput(BaseModel):
    fecha_inicio: str
    fecha_fin: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_rf")
async def predict_rf(input_data: RandomForestInput):
    try:
        # Convertir la fecha en año, mes y día
        fecha = datetime.strptime(input_data.fecha, '%Y-%m-%d')
        anio = fecha.year
        mes = fecha.month
        dia = fecha.day

        # Preparar los datos para el modelo
        input_df = pd.DataFrame([{
            "anio": anio,
            "mes": mes,
            "dia": dia,
            "hora": input_data.hora,
            "codigo_localidad": input_data.codigo_localidad,
            "edad": input_data.edad,
            "genero": input_data.genero,
            "tipo_incidente": input_data.tipo_incidente
        }])

        prediction = rf_model.predict(input_df)
        result = int(prediction[0])
        return JSONResponse(content={'prediction': result})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.post("/predict_sarimax")
async def predict_sarimax(input_data: SarimaxInput):
    try:
        fecha_inicio = datetime.strptime(input_data.fecha_inicio, '%Y-%m-%d')
        fecha_fin = datetime.strptime(input_data.fecha_fin, '%Y-%m-%d')
        
        date_range = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        
        model_sarimax_fit = model_sarimax.fit(disp=False)
        prediction = model_sarimax_fit.predict(start=date_range[0], end=date_range[-1])
        result = [{"date": date.strftime('%Y-%m-%d'), "prediction": pred} for date, pred in zip(date_range, prediction)]
        
        return JSONResponse(content={'prediction': result})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
