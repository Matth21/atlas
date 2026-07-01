from typing import Callable

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from atlas.api.auth import AuthError, verify_clerk_jwt
from atlas.api.free_tier import FREE_TIER_CONFIG, FreeTierError, validate_free_tier_request
from atlas.api.jobs import JobStore


class CompressRequest(BaseModel):
    model_id: str
    config: dict


def create_app(
    job_store: JobStore,
    jwks_client,
    clerk_issuer: str,
    enqueue: Callable[[str], None],
) -> FastAPI:
    app = FastAPI()

    @app.post("/compress", status_code=202)
    def compress(request: CompressRequest, authorization: str | None = Header(default=None)):
        user_id: str | None = None
        if authorization is not None:
            token = authorization.removeprefix("Bearer ").strip()
            try:
                user_id = verify_clerk_jwt(token, jwks_client, clerk_issuer)
            except AuthError as e:
                raise HTTPException(status_code=402, detail=str(e)) from e

        if user_id is None:
            if request.config != FREE_TIER_CONFIG:
                # A non-default config is a pro-tier feature: without a valid
                # token there is nothing to evaluate against the free tier,
                # so this is "payment required", not "forbidden".
                raise HTTPException(
                    status_code=402,
                    detail="custom config requires Pro tier authentication",
                )
            try:
                validate_free_tier_request(request.model_id, request.config)
            except FreeTierError as e:
                raise HTTPException(status_code=403, detail=str(e)) from e
            tier = "free"
        else:
            tier = "pro"

        job = job_store.create(
            model_id=request.model_id,
            tier=tier,
            config=request.config,
            user_id=user_id,
        )
        enqueue(job.id)
        return {"job_id": job.id, "status": job.status}

    @app.get("/compress/{job_id}")
    def get_compress(job_id: str):
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "result": job.result,
            "error": job.error,
        }

    return app
