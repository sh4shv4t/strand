from dotenv import load_dotenv

# Runs before any submodule is imported, so env vars from .env (see
# .env.example) are set before anything reads os.environ -- e.g.
# vector_store.DenseScorer checks STRAND_DISABLE_EMBEDDINGS at import time.
load_dotenv()
