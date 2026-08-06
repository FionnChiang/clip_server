from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query

from ..document_converter import is_document, document_to_images
from ..schemas.inference import (
    DocumentPredictionResult, PagePredictionResult,
    DocumentTopKResult, PageTopKResult,
)
from ..services import inference_client

router = APIRouter(prefix="/api/projects/{project_id}", tags=["inference"])


@router.post("/predict")
async def predict(
    project_id: str,
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    margin_threshold: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
):
    contents = await file.read()
    try:
        if is_document(file.filename):
            pages = document_to_images(file.filename, contents)
            page_results = await inference_client.predict_pages(
                project_id, pages, model_id, top_k=1,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
                temperature=temperature,
            )
            return DocumentPredictionResult(
                filename=file.filename,
                page_count=len(pages),
                results=[
                    PagePredictionResult(page=page_no, **results[0].model_dump())
                    for page_no, results in page_results
                    if results
                ],
            )
        results = await inference_client.predict(
            project_id, contents, model_id, top_k=1,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            temperature=temperature,
        )
        return results[0] if results else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/predict/top-k")
async def predict_top_k(
    project_id: str,
    file: UploadFile = File(...),
    k: int = Query(3, ge=1, le=10),
    model_id: Optional[str] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    margin_threshold: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
):
    contents = await file.read()
    try:
        if is_document(file.filename):
            pages = document_to_images(file.filename, contents)
            page_results = await inference_client.predict_pages(
                project_id, pages, model_id, top_k=k,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
                temperature=temperature,
            )
            return DocumentTopKResult(
                filename=file.filename,
                page_count=len(pages),
                results=[
                    PageTopKResult(
                        page=page_no,
                        top_k=results,
                    )
                    for page_no, results in page_results
                ],
            )
        results = await inference_client.predict(
            project_id, contents, model_id, top_k=k,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            temperature=temperature,
        )
        return {"results": [r.model_dump() for r in results]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
