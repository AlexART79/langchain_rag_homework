from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_ollama import ChatOllama

from index_data import retriever

def run_llm(query: str, chat_history: List[BaseMessage] = []):
    llm = ChatOllama(
        model="gemma3:latest",
        temperature=0,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=rephrase_prompt)

    retrieval_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_chain)
    response = retrieval_chain.invoke(input={"input": query, "chat_history": chat_history})

    ret_val = {
        "query": response["input"],
        "result": response["answer"],
        "source_documents": response["context"],
    }
    return ret_val

if __name__ == "__main__":

    chat_history: List[BaseMessage] = []

    while True:
        print("Wellcome! Input your question. Type 'exit' to exit")
        query = input(">> ")

        if query == 'exit':
            break

        res = run_llm(query=query, chat_history=chat_history)

        print(res["result"])

        chat_history.append(HumanMessage(query))
        chat_history.append(AIMessage(res["result"]))