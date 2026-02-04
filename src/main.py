import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.pdf import PDFExtractor
from src.vectordb import VectorStore
from src.rag import RAGPipeline

console = Console()


def parse_servers(server_cmds: tuple) -> list[dict]:
    servers = []
    for i, cmd in enumerate(server_cmds):
        parts = cmd.split()
        servers.append({
            "name": f"server_{i}",
            "command": parts[0],
            "args": parts[1:] if len(parts) > 1 else [],
        })
    return servers


@click.group()
def cli():
    """AI Research Assistant for Technical PDFs"""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def ingest(pdf_path: str):
    """Ingest a PDF document into the knowledge base."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    path = Path(pdf_path)
    extractor = PDFExtractor()
    store = VectorStore()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting text...", total=100)
        chunks = extractor.extract_chunks(path)
        progress.update(task, completed=50, description="Generating embeddings...")
        store.add_chunks(chunks, progress_callback=lambda p: progress.update(task, completed=50 + p * 50))
        progress.update(task, completed=100, description="Done")

    console.print(f"[green]✓[/green] Ingested {len(chunks)} chunks from {path.name}")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
def search(query: str, top_k: int):
    """Search for relevant content in documents."""
    store = VectorStore()
    results = store.search(query, top_k)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(
            Panel(
                r["text"][:500],
                title=f"[{i}] {r['metadata']['source_file']} (Page {r['metadata']['page_number']})",
            )
        )


@cli.command()
@click.argument("question")
def ask(question: str):
    """Ask a question about the documents (simple RAG)."""
    pipeline = RAGPipeline()
    with console.status("Thinking..."):
        answer = pipeline.query(question)
    console.print(Panel(answer, title="Answer"))


@cli.command()
@click.argument("user_query")
@click.option("--server", "-s", multiple=True, help="MCP server command")
def query(user_query: str, server: tuple):
    """Smart query with agent routing and MCP tools."""
    asyncio.run(_query_async(user_query, server))


async def _query_async(user_query: str, server: tuple):
    from src.mcp.client import MCPClient
    from src.agents import AgentRouter

    servers = parse_servers(server)
    mcp_client = None

    if servers:
        mcp_client = MCPClient()
        for s in servers:
            await mcp_client.connect(s["name"], s["command"], s.get("args", []))
        tools = mcp_client.get_openai_tools()
        console.print(f"[dim]Connected. Tools: {[t['function']['name'] for t in tools]}[/dim]")

    try:
        router = AgentRouter(mcp_client)
        with console.status("Processing..."):
            answer = await router.route_async(user_query)
        console.print(Panel(answer, title="Response"))
    finally:
        if mcp_client:
            await mcp_client.close()


@cli.command()
def documents():
    """List all ingested documents."""
    store = VectorStore()
    docs = store.list_documents()

    if not docs:
        console.print("[yellow]No documents ingested[/yellow]")
        return

    for doc in docs:
        console.print(f"  • {doc}")


@cli.command()
def mcp():
    """Start the MCP server."""
    from src.mcp.server import run_server
    asyncio.run(run_server())


@cli.command()
@click.option("--server", "-s", multiple=True, help="MCP server command")
def interactive(server: tuple):
    """Start interactive chat mode with MCP tools."""
    asyncio.run(_interactive_async(server))


async def _interactive_async(server: tuple):
    from src.mcp.client import MCPClient
    from src.agents.specialized import QAAgent

    servers = parse_servers(server)
    mcp_client = None

    if servers:
        mcp_client = MCPClient()
        for s in servers:
            await mcp_client.connect(s["name"], s["command"], s.get("args", []))

    console.print("[bold]AI Research Assistant[/bold]")
    if mcp_client:
        tools = mcp_client.get_openai_tools()
        console.print(f"[dim]Connected. Tools: {[t['function']['name'] for t in tools]}[/dim]")
    console.print("Type 'quit' to exit\n")

    try:
        while True:
            try:
                user_query = console.input("[bold blue]You:[/bold blue] ")
                if user_query.lower() in ("quit", "exit", "q"):
                    break

                agent = QAAgent(mcp_client)
                with console.status("Thinking..."):
                    answer = await agent.process_async(user_query)

                console.print(f"\n[bold green]Assistant:[/bold green] {answer}\n")
            except KeyboardInterrupt:
                break
    finally:
        if mcp_client:
            await mcp_client.close()

    console.print("\n[dim]Goodbye![/dim]")


def main():
    cli()


if __name__ == "__main__":
    main()
