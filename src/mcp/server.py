import logging
import sys
import traceback
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.pdf import PDFExtractor
from src.vectordb import VectorStore
from src.rag import RAGPipeline

logger = logging.getLogger(__name__)


def create_mcp_server() -> Server:
    server = Server("ai-research-assistant")
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store)
    pdf_extractor = PDFExtractor()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="ingest_pdf",
                description="Ingest a PDF document into the knowledge base. May take time for large files.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the PDF file"}
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="search_documents",
                description="Search for relevant content in ingested documents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="ask_question",
                description="Ask a question and get an answer based on ingested documents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Your question"}
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="list_documents",
                description="List all ingested documents",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        logger.info(f"Tool: {name} | Args: {arguments}")

        try:
            if name == "ingest_pdf":
                path = Path(arguments["file_path"])
                if not path.exists():
                    return [TextContent(type="text", text=f"File not found: {path}")]

                logger.info(f"[ingest] Extracting: {path.name}")
                chunks = pdf_extractor.extract_chunks(path)
                logger.info(f"[ingest] Got {len(chunks)} chunks, embedding...")

                vector_store.add_chunks(chunks)
                logger.info(f"[ingest] Done: {len(chunks)} chunks from {path.name}")

                return [
                    TextContent(
                        type="text",
                        text=f"Ingested {len(chunks)} chunks from {path.name}",
                    )
                ]

            elif name == "search_documents":
                query = arguments["query"]
                logger.info(f"[search] Query: {query[:50]}...")
                results = vector_store.search(query, arguments.get("top_k", 5))

                if not results:
                    return [TextContent(type="text", text="No results found")]

                output = []
                for i, r in enumerate(results, 1):
                    output.append(
                        f"{i}. [{r['metadata']['source_file']} p.{r['metadata']['page_number']}] {r['text'][:200]}..."
                    )
                logger.info(f"[search] Found {len(results)} results")
                return [TextContent(type="text", text="\n\n".join(output))]

            elif name == "ask_question":
                question = arguments["question"]
                logger.info(f"[ask] Question: {question[:50]}...")
                answer = rag_pipeline.query(question)
                logger.info(f"[ask] Answer: {len(answer)} chars")
                return [TextContent(type="text", text=answer)]

            elif name == "list_documents":
                docs = vector_store.list_documents()
                logger.info(f"[list] {len(docs)} documents")
                if not docs:
                    return [TextContent(type="text", text="No documents ingested")]
                return [TextContent(type="text", text="\n".join(docs))]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Tool error: {e}\n{traceback.format_exc()}")
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


async def run_server():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting MCP server...")
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
