from src.ai import create_ai
from src.llm import get_llm
from src.ui import spinner, chat_loop

def main():

    # llm = get_llm()
    # ai = create_ai(llm)
    # ChatApp(ai).run()
    with spinner("building..."):
        llm = get_llm()
        ai = create_ai(llm)
        chain = ai.build_chain()
    chat_loop(chain, ai.routes)
