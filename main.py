import subprocess
import uuid
import ollama
from langchain_community.chat_models import ChatOllama  # Using Ollama for Local LLM
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated

# Initialize Ollama LLM
llm = ChatOllama(model="llama3.2")  # Change to "llama3" or any model you prefer


# Function to execute shell commands
def run_shell_command(command: str) -> str:
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip() if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


# Convert Natural Language to Shell Command using Ollama LLM
def convert_to_shell_command(natural_language: str) -> str:
    """Uses Ollama's LLM to convert natural language input into a shell command."""
    prompt = f"""
    Convert the following natural language request into a valid shell command.
    Return ONLY the shell command without any explanations.
    
    Example:
    - Input: "List all running containers"
    - Output: "podman ps"

    - Input: "Show disk space usage"
    - Output: "df -h"

    Now convert this request: "{natural_language}"
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()  # Extract command from LLM response


# Process shell command (with Ollama)
def process_shell_tool(state):
    """Executes shell command from user input after LLM conversion."""
    natural_language = state["messages"][-1].content.strip()
    
    # Convert user input to shell command
    command = convert_to_shell_command(natural_language)
    
    # Execute the shell command
    output = run_shell_command(command)
    
    return {"messages": [AIMessage(content=f"Command: {command}\n\nOutput:\n{output}")]}



# Define AI Agent State
class State(TypedDict):
    messages: Annotated[list, ...]


# Setup Workflow
memory = MemorySaver()
workflow = StateGraph(State)

# Add Nodes (Processing the shell command)
workflow.add_node("process_shell_tool", process_shell_tool)

# Define Transitions (No Conditional Edges)
workflow.add_edge(START, "process_shell_tool")
workflow.add_edge("process_shell_tool", END)

# Compile Workflow
graph = workflow.compile(checkpointer=memory)

# Configuration
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# **User Input Execution**
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in {"q", "quit", "exit"}:
        print("Goodbye!")
        break

    output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    print("AI:", output["messages"][-1].content)