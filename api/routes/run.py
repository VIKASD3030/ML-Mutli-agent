"""
api/routes/runs.py

POST /runs           — create and queue a run (structured spec OR
                        context+file via RequirementAgent)
GET  /runs/{run_id}   — poll one run's current full state
GET  /runs            — list runs, lightweight
"""

from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.requirement_agent import RequirementAgent
from api.schemas_api import (
    ClarificationResponse,
    RunCreatedResponse,
    RunCreateRequest,
    RunDetail,
    RunSummary,
)

from db.models import RunRecord
from db.session import SessionLocal, get_db
from db.sync import load_run_state, persist_run_state
from pipeline.run_pipeline import run as run_pipeline
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ClarificationNeeded, ProblemSpec

router = APIRouter(prefix="/runs", tags=["runs"])

def _execute_run_in_background(run_id: str, spec: ProblemSpec) -> None:
    db= SessionLocal()
    try:
        def on_update(pipeline_run: PipelineRun) -> None:
            persist_run_state(db, pipeline_run)

        run_pipeline(spec, run_id=run_id, on_update=on_update)
    except Exception:
        run_record = db.get(RunRecord, run_id)
        if run_record is not None:
            run_record.status = "escalated"
            db.commit()
    finally:
        db.close()

@router.post("", response_model=Union[RunCreatedResponse, ClarificationResponse])
def create_run(
    request: RunCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if request.problem_spec is not None:
        spec = request.problem_spec
    else:
        agent = RequirementAgent()
        result = agent.run(request.context, file_path=request.file_path)

        if isinstance(result, ClarificationNeeded):
            return ClarificationResponse(
                missing_fields= result.missing_fields,
                question_for_user=result.question_for_user,
            )
        spec = result

    run_id = str(uuid.uuid4())

    persist_run_state(db, PipelineRun(run_id=run_id, problem_spec=spec))

    background_tasks.add_task(_execute_run_in_background, run_id, spec)

    return RunCreatedResponse(run_id=run_id)

@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, db: Session = Depends(get_db)):
    pipeline_run = load_run_state(db, run_id)
    if pipeline_run is None:
        raise HTTPException(status_code=404, detail=f"No run found with id {run_id!r}")
    return RunDetail.from_pipeline_run(pipeline_run)

@router.get("", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)):
    records = db.execute(select(RunRecord).order_by(RunRecord.created_at.desc())).scalars().all()
    return [
        RunSummary(
            run_id=r.run_id, status=r.status,
            task_type=r.problem_spec.task_type if r.problem_spec else None,
            pass_fail=r.evaluation_report.pass_fail if r.evaluation_report else None,
            metric_value=r.evaluation_report.metric_value if r.evaluation_report else None,
            loop_count=r.loop_count, created_at=r.created_at, updated_at=r.updated_at,
        )
        for r in records
    ]
