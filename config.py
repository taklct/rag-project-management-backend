# connection
ENDPOINT_URL ="https://taklc-mfcql77l-eastus2.cognitiveservices.azure.com/"
DEPLOYMENT_NAME = "gpt-5-mini"
AZURE_OPENAI_API_KEY="7hOTlHxbEM2xUCxTrA8Oziobn1Sat9pWeW9dWjH44RYDxlf4t6zYJQQJ99BIACHYHv6XJ3w3AAAAACOGCmaY"
EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
JIRA_ENDPOINT_URL="https://origat.atlassian.net"
JIRA_API_URL=f"{JIRA_ENDPOINT_URL}/rest/api/3"
JIRA_USERNAME="taklct113@gmail.com"
JIRA_API_TOKEN="ATATT3xFfGF0YA8k3ja2irnXqdEzpHV1PLJWeTy1PW0JuQD264SIB-_4wSKOmfI0jY2u5RkZelbAlTPzSN9Hw8CR_lXIJzI9KE8Wm8WabmVWtFGCNq7CDcXL-G0KYrxqCFmVrQKoUCudTidQFlSUUJdZKmaeptOjk-Tmz9Cm1O34BM1UuL_xSi0=C881598E"
# Optional override for Azure OpenAI API version
API_VERSION = "2025-01-01-preview"

# openAI config
DEFAULT_TOP_K = 10
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_COMPLETION_TOKENS = 10000

# Paths
SOURCE_DIR = "./data_sources"
LOG_DIR = "./logs"
LOG_PATH = f"{LOG_DIR}/query_logs.csv"
