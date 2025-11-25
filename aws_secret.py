import boto3
import json

SECRET_ID = "labs/rag-project-management"
REGION = "ap-southeast-1"

def get_secret():
    client = boto3.client("secretsmanager", region_name=REGION)

    response = client.get_secret_value(SecretId=SECRET_ID)

    if "SecretString" in response:
        secret = response["SecretString"]
    else:
        # Binary secrets (rare)
        import base64
        secret = base64.b64decode(response["SecretBinary"])

    # Convert JSON string → dict
    return json.loads(secret)

# Usage example
secret_value = get_secret()