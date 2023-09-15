from src.ai import create_ai
from src.ui import spinner, chat_loop

def main():

    with spinner("building..."):
        ai = create_ai()
        chain = ai.build_chain()
    chat_loop(chain, ai.routes)


