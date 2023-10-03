import typer
from src.vectore_store import Coordinator
from src.models import RAG, AppSettings

app = typer.Typer()

@app.command(name="create")
def create(name: str, description: str, directory: str, persist_directory: str):
    settings = AppSettings()
    persist_dir = f"{persist_directory}/{settings.vector_storage_subdir}"
    Coordinator().load_or_create(directory, persist_dir)
    rag = RAG(name=name, description=description, directory=directory, persist_directory=persist_dir)
    settings.rags.append(rag)
    settings.save()


@app.command(name="refresh")
def refresh(name: str):
    settings = AppSettings()
    for r in settings.rags:
        if r.name == name:
            Coordinator().load_or_create(r.directory, r.persist_directory, refresh=True)


@app.command(name="destroy")
def destroy(name: str):
    settings = AppSettings()
    for r in settings.rags:
        if r.name == name:
            Coordinator().destroy(r.persist_directory)
    settings.rags = [r for r in settings.rags if r.name != name]
    settings.save()