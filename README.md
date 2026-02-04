# query with rag
uv run python -m src.main query "What is attention" -s "uv run python -m src.main mcp"
                                                                                                
# (simple QA)                                               
uv run python -m src.main ask "What is attention" 

# Search
uv run python -m src.main search "attention mechanism" -k 5

# interactive
uv run python -m src.main interactive -s "uv run python -m src.main mcp"

# ingest
uv run python -m src.main ingest /path/to/file.pdf

# list 
uv run python -m src.main documents
