import typer
from src.commands import chat, rag
from src.commands.rag import app as rag_commands

app = typer.Typer()

app.command(name="chat")(chat.main)

app.add_typer(rag_commands, name="rag")
    
if __name__ == "__main__":
    app()