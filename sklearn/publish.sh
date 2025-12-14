# Fix: Remove /serverless-lambda from ECR_URL
ECR_URL=906063354528.dkr.ecr.eu-north-1.amazonaws.com
REPO_NAME=serverless-lambda
LOCAL_IMAGE=serverless-lambda

docker build -t ${LOCAL_IMAGE} .

aws ecr get-login-password \
  --region "eu-north-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

# Fix: Use correct path
REMOTE_IMAGE_TAG="${ECR_URL}/${REPO_NAME}:v1"

docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}