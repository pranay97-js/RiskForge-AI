"""FastAPI Backend Service for RiskForge AI (Re-exported from app.py)."""

from app import (
    ActionRequest,
    CostMatrixRequest,
    EvaluationResponse,
    EvidencePayload,
    RepresentmentResponse,
    TransactionPayload,
    app,
    get_ai_responder,
    get_decision_service,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
