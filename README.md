# Chan + KNN Stock Backend

FastAPI backend for A-share daily analysis that combines:

- Simplified Chan theory structure extraction (fractal, bi, segment)
- KNN classifier for next-day direction prediction

## Quick start

1. Create/activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open docs:

   - <http://127.0.0.1:8000/docs>

## Main endpoints

- `GET /health`
- `POST /v1/data/sync`
- `POST /v1/model/train`
- `GET /v1/model/metrics`
- `GET /v1/analyze/{symbol}`

## Notes

- Default data source is Tushare (`pro.daily`), configurable via `STOCK_AI_DATA_PROVIDER`.
- Cached market data and model files are stored under `backend/storage/`.
- This implementation is an MVP and focuses on reliable engineering structure first.

## Data source config

- `STOCK_AI_DATA_PROVIDER=auto` (default), `tushare` or `akshare`
- `STOCK_AI_TUSHARE_TOKEN=your_token` when using tushare
# stock_backend
# stock_backend
