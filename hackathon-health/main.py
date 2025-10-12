from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from twilio.rest import Client
import os
import pandas as pd

from predict import make_prediction

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Load datasets
inventory_df = pd.read_csv(BASE_DIR / "inventory_dataset.csv")
inventory_df["last_restock_date"] = pd.to_datetime(inventory_df["last_restock_date"], errors="coerce")

# we will use 50 facilities for a proof of concept
ALL_FACILITIES = sorted(inventory_df["facility_id"].dropna().unique().tolist())
TOP_FACILITIES = ALL_FACILITIES[:50]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "facilities": TOP_FACILITIES}
    )

# API: Get all items for a facility
@app.get("/api/items/{facility_id}")
async def get_items_by_facility(facility_id: str):
    items = inventory_df[inventory_df["facility_id"] == facility_id]
    if items.empty:
        return {"error": f"No items found for {facility_id}"}
    return {
        "items": items[["item_name", "stock_level", "reorder_level", "last_restock_date"]]
        .fillna("")
        .to_dict(orient="records")
    }

# API: Get details for one item
@app.get("/api/item-details/{facility_id}/{item_name}")
async def get_item_details(facility_id: str, item_name: str):
    row = inventory_df[
        (inventory_df["facility_id"] == facility_id) &
        (inventory_df["item_name"] == item_name)
    ]
    if row.empty:
        return {"error": "Item not found"}
    r = row.iloc[0]
    return {
        "stock_level": float(r["stock_level"]),
        "reorder_level": float(r["reorder_level"]),
        "last_restock_date": r["last_restock_date"].strftime("%Y-%m-%d") if pd.notna(r["last_restock_date"]) else "2024-01-01"
    }

# ML Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    facility_id: str = Form(...),
    item_name: str = Form(...),
    current_stock_level: float = Form(...),
    reorder_level: float = Form(...),
    last_restock_date: str = Form(...)
):
    try:
        result = make_prediction(
            facility_id=facility_id,
            item_name=item_name,
            current_stock_level=current_stock_level,
            reorder_level=reorder_level,
            last_restock_date=last_restock_date
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "facilities": TOP_FACILITIES,
                "prediction": result,
                "inputs": {
                    "facility_id": facility_id,
                    "item_name": item_name,
                    "current_stock_level": current_stock_level,
                    "reorder_level": reorder_level,
                    "last_restock_date": last_restock_date
                }
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "facilities": TOP_FACILITIES,
                "error": str(e),
                "inputs": {
                    "facility_id": facility_id,
                    "item_name": item_name,
                    "current_stock_level": current_stock_level,
                    "reorder_level": reorder_level,
                    "last_restock_date": last_restock_date
                }
            }
        )