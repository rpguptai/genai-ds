"""
agents.py - Defines the specialized agents for the multi-agent system.

Author: Ravi
Date: 2023-10-27

This file defines three distinct agent types:
1. A Local Researcher for searching documents.
2. A Web Researcher for searching the internet.
3. A Writer for synthesizing the final answer.
"""

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, Tool
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# --- Agent 1: Local Document Researcher ---

@tool
def search_local_documents(query: str, rag_service) -> str:
    """Searches for information specifically within the user's uploaded PDF documents."""
    retriever = rag_service.vector_store.as_retriever()
    # The correct method is .invoke() in modern LangChain
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No information found in the local documents for that query."

def create_local_researcher(llm_name: str, rag_service) -> AgentExecutor:
    """Creates an agent that only searches local documents."""
    local_search_tool = Tool(
        name=search_local_documents.name,
        func=lambda q: search_local_documents.func(q, rag_service=rag_service),
        description=search_local_documents.description
    )
    tools = [local_search_tool]
    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Your only job is to use the `search_local_documents` tool to find information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Agent 2: Web Researcher ---

@tool
def search_web(query: str) -> str:
    """Searches the web, specifically google.com and bing.com, for information."""
    return f"Simulated search results from Google and Bing for: '{query}'. The web says hello."

def create_web_researcher(llm_name: str) -> AgentExecutor:
    """Creates an agent that only searches the web."""
    tools = [search_web]
    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Your only job is to use the `search_web` tool to find information on google.com and bing.com."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Agent 4: Writer ---

def create_writer_agent(llm_name: str) -> Runnable:
    """Creates the agent responsible for synthesizing the final answer."""
    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert writer. Your job is to synthesize the provided research findings from local and web searches into a clear, comprehensive answer to the user's question. Provide only the final answer."),
        ("human", "Question: {question}\n\nResearch Findings:\n{context}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain
