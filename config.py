from pathlib import Path

EXPERT_MODEL = "gemini-2.5-pro"
STORYTELLER_MODEL = "claude-opus-4-7"

FILE_SEARCH_MAX_FILE_MB = 100
FILE_SEARCH_FREE_TIER_GB = 1

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

HANDOFF_CONTEXT_TURNS = 10

INDEXING_POLL_INTERVAL_SEC = 2
INDEXING_TIMEOUT_SEC = 180

PUBMED_DEFAULT_RESULTS = 5
PUBMED_RATE_LIMIT_NO_KEY = 3
PUBMED_RATE_LIMIT_WITH_KEY = 10

DRIVE_TOKEN_PATH = Path.home() / ".lm-app" / "drive_token.json"
DRIVE_CREDENTIALS_PATH = Path.home() / ".lm-app" / "credentials.json"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
