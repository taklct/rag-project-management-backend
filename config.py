# connection
ENDPOINT_URL ="https://taklc-mfcql77l-eastus2.cognitiveservices.azure.com/"
DEPLOYMENT_NAME = "gpt-5-mini"
AZURE_OPENAI_API_KEY="7hOTlHxbEM2xUCxTrA8Oziobn1Sat9pWeW9dWjH44RYDxlf4t6zYJQQJ99BIACHYHv6XJ3w3AAAAACOGCmaY"
EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
JIRA_ENDPOINT_URL="https://origat.atlassian.net"
JIRA_API_URL=f"{JIRA_ENDPOINT_URL}/rest/api/3"
JIRA_USERNAME="taklct113@gmail.com"
JIRA_API_TOKEN="ATATT3xFfGF0Ce7-Az_Xb81T5M47fU2Z3wNKkNaawjbjBaF9rkltnW_sqSvYibhXWTW2_onOpQYoveyFkh_3aszvfFajy3EcK1MDyAxK6l_1nQDLxAHCa9-HTao9QJ4zkrPG5OWKrHO8A_5iBBnMHSxUTsYH4jrxeUhOIywuz0vZKnmn97oqo9E=70439450"
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
