"""Token Vending Machine - Issues scoped AWS credentials based on Kubernetes service account identity."""

import logging
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated

import boto3
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from jinja2 import Template, TemplateError
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pythonjsonlogger import jsonlogger

# Context variable for request-scoped data
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class ContextFilter(logging.Filter):
    """Logging filter that adds request context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured JSON logging to stdout."""
    logger = logging.getLogger("tvm")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    aws_role_arn: str
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    policy_template_path: str = "/etc/token-vending-machine/policy.json"
    session_duration_seconds: int = 3600
    log_level: str = "INFO"
    port: int = 8000

    class Config:
        env_file = ".env"


class AwsCredentials(BaseModel):
    """AWS credentials response model."""

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_SESSION_TOKEN: str


# Initialize settings and logger at startup
# Settings are loaded from environment variables by pydantic-settings
settings = Settings()  # type: ignore[call-arg]
logger = setup_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    logger.info(
        "Token Vending Machine starting",
        extra={
            "aws_region": settings.aws_region,
            "aws_role_arn": settings.aws_role_arn,
            "policy_template_path": settings.policy_template_path,
            "session_duration_seconds": settings.session_duration_seconds,
        },
    )
    yield
    logger.info("Token Vending Machine shutting down")


def get_settings() -> Settings:
    """Dependency to get application settings."""
    return settings


def get_k8s_client() -> client.AuthenticationV1Api:
    """Dependency to get Kubernetes authentication API client."""
    config.load_incluster_config()
    return client.AuthenticationV1Api()


def get_sts_client(settings: Annotated[Settings, Depends(get_settings)]) -> boto3.client:
    """Dependency to get AWS STS client."""
    return boto3.client(
        "sts",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


app = FastAPI(
    title="Token Vending Machine",
    description="Issues scoped AWS credentials based on Kubernetes service account identity",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log requests and add request ID context."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_ctx.set(request_id)

    start_time = time.perf_counter()

    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Request failed with exception",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
                "error": str(e),
            },
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Log HTTP exceptions."""
    logger.warning(
        "HTTP error response",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"X-Request-ID": request_id_ctx.get()},
    )


def extract_bearer_token(authorization: str = Header(...)) -> str:
    """Extract and validate Bearer token from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'",
        )
    return authorization[7:]  # Remove "Bearer " prefix


def validate_token_and_extract_identity(
    token: str,
    k8s_client: client.AuthenticationV1Api,
) -> tuple[str, str]:
    """
    Validate a Kubernetes service account token and extract identity.

    Returns:
        Tuple of (namespace, service_account_name)
    """
    token_review = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token=token)
    )

    try:
        response = k8s_client.create_token_review(token_review)
    except ApiException as e:
        logger.error(
            "Kubernetes API error during token validation",
            extra={"error": e.reason, "status": e.status},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Failed to validate token: {e.reason}",
        )

    if not response.status.authenticated:
        logger.warning("Token authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not valid or has expired",
        )

    # Username format: system:serviceaccount:<namespace>:<service-account-name>
    username = response.status.user.username
    if not username.startswith("system:serviceaccount:"):
        logger.warning(
            "Token is not from a service account",
            extra={"username": username},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not from a service account",
        )

    parts = username.split(":")
    if len(parts) != 4:
        logger.warning(
            "Invalid service account token format",
            extra={"username": username},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service account token format",
        )

    namespace = parts[2]
    service_account = parts[3]

    logger.info(
        "Token validated successfully",
        extra={"namespace": namespace, "service_account": service_account},
    )

    return namespace, service_account


def load_and_render_policy(
    template_path: str,
    namespace: str,
    service_account: str,
) -> str:
    """Load IAM policy template and render with Jinja2."""
    try:
        policy_template = Path(template_path).read_text()
    except FileNotFoundError:
        logger.error("Policy template not found", extra={"path": template_path})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Policy template not found",
        )
    except OSError as e:
        logger.error(
            "Failed to read policy template",
            extra={"path": template_path, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read policy template: {e}",
        )

    # Render the template with Jinja2
    try:
        rendered_policy = Template(policy_template).render(
            namespace=namespace,
            serviceaccount=service_account,
        )
    except TemplateError as e:
        logger.error(
            "Failed to render policy template",
            extra={"path": template_path, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render policy template: {e}",
        )

    logger.debug(
        "Policy template rendered",
        extra={"namespace": namespace, "service_account": service_account},
    )

    return rendered_policy


def assume_role_with_policy(
    sts_client: boto3.client,
    role_arn: str,
    session_name: str,
    policy: str,
    duration_seconds: int,
) -> AwsCredentials:
    """Assume an AWS role with the given inline policy."""
    try:
        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            Policy=policy,
            DurationSeconds=duration_seconds,
        )
    except Exception as e:  # noqa: BLE001 - normalize SDK failures to an HTTP response
        logger.error(
            "Failed to assume AWS role",
            extra={"role_arn": role_arn, "session_name": session_name, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assume role: {e}",
        )

    logger.info(
        "AWS role assumed successfully",
        extra={"role_arn": role_arn, "session_name": session_name},
    )

    credentials = response["Credentials"]
    return AwsCredentials(
        AWS_ACCESS_KEY_ID=credentials["AccessKeyId"],
        AWS_SECRET_ACCESS_KEY=credentials["SecretAccessKey"],
        AWS_SESSION_TOKEN=credentials["SessionToken"],
    )


@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/credentials", response_model=AwsCredentials)
def get_credentials(
    token: Annotated[str, Depends(extract_bearer_token)],
    settings: Annotated[Settings, Depends(get_settings)],
    k8s_client: Annotated[client.AuthenticationV1Api, Depends(get_k8s_client)],
    sts_client: Annotated[boto3.client, Depends(get_sts_client)],
) -> AwsCredentials:
    """
    Exchange a Kubernetes service account token for scoped AWS credentials.

    The token is validated against the Kubernetes API, and the service account's
    namespace and name are used to render an IAM policy template. The rendered
    policy is then used to assume an AWS role with scoped permissions.
    """
    # Validate token and extract identity
    namespace, service_account = validate_token_and_extract_identity(token, k8s_client)

    # Load and render the IAM policy template
    policy = load_and_render_policy(
        settings.policy_template_path,
        namespace,
        service_account,
    )

    # Generate a session name from the identity
    session_name = f"{namespace}-{service_account}"[:64]  # AWS limits to 64 chars

    # Assume the role with the scoped policy
    return assume_role_with_policy(
        sts_client,
        settings.aws_role_arn,
        session_name,
        policy,
        settings.session_duration_seconds,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
