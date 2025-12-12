FROM public.ecr.aws/lambda/python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN pip install uv
COPY pyproject.toml uv.lock ./

# RUN uv sync
RUN uv pip install --system -r <(uv export --format requirements-txt)

COPY model_C=1.0.bin .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]