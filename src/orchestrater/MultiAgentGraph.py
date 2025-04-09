"""
MultiAgentGraph.py

This module defines the multi-agent graph for the generative AI project.
It coordinates different agent nodes (Insights Agent, Cost Saving Agent, etc.)
using a supervisor to route the conversation flow.
Prompt templates are loaded from the prompt_templates folder.
"""

import os
import re
import uuid
import pandas as pd
import streamlit as st
import functools
import warnings
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import operator

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain.schema import HumanMessage

from config import display_saved_plot
from src.agents.BIAgent_Node import BIAgent_Class, execute_analysis
from src.orchestrater.supervisor import supervisor_chain, members
from src.agents.CostOptimization_Node import AgenticCostOptimizer
from src.agents.Static_CostOptimization_Node import Static_CostOptimization_Class
from src.utils.openai_api import get_supervisor_llm
from src.utils.load_templates import load_template
from src.core.order_consolidation.consolidation_ui import show_ui_cost_saving_agent, show_ui_cost_saving_agent_static

warnings.filterwarnings("ignore")

# Load environment variables
_ = load_dotenv(find_dotenv())

llm = get_supervisor_llm()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str

def get_question(state, supervisor_chain):
    """
    Extract a single-line question from the conversation history to pass to an agent.
    """
    all_msg = []
    for msg in state['messages']:
        all_msg.append(msg.content)

    text = f"""Next agent to be called, use this conversation: {"\n".join(all_msg)} to create a single 
    line question to be passed next to this agent as a question. Keep track of these parameters while summarising:
        start_date;
        end_date;
        group_method;
        all_post_code; 
        all_customers;
        selected_postcodes; 
        selected_customers;
        scenario;
        shipment_window_range;
        total_shipment_capacity;
        utilization_threshold;
        .Provide your answer in `direct_response`."""
    response = supervisor_chain.invoke([HumanMessage(content=text)])
    print("Inside Get Question; Direct_response\n", response['direct_response'])
    return response

def supervisor_node(state: AgentState):
    """
    Supervisor Node: Uses the supervisor chain to determine the next agent.
    Also handles direct responses from the supervisor.
    """
    result = supervisor_chain.invoke(state['messages'])
    
    # Handle direct responses from the supervisor
    if result['next'] == 'SELF_RESPONSE':
        if 'direct_response' in result:
            return {"messages": [AIMessage(content=result['direct_response'])], "next": "FINISH"}
        else:
            # Fallback if direct_response field is somehow missing
            return {"messages": [AIMessage(content="I understand your question. Let me answer directly.")], "next": "FINISH"}
    
    # Original routing logic - use thought_process if available, otherwise use a generic message
    thought_process = result.get('thought_process', f"Calling {result['next']}...")
    return {"messages": [AIMessage(content=thought_process)], "next": result['next']}

# ---------------------- Generic Agent Node ----------------------

def agent_node(state, agent, name):
    """
    Generic agent node that calls the provided agent function with the state.
    """
    result = agent(state)
    return {"messages": result, "next": "supervisor"}

# ---------------------- Insights Agent ----------------------

def bi_agent(state: AgentState):
    """
    Insights Agent is responsible for analyzing shipment data to generate insights. 
    It handles tasks such as performing exploratory data analysis (EDA), calculating summary statistics, identifying trends, 
    comparing metrics across different dimensions (e.g., users, regions), and generating visualizations to help 
    understand shipment-related patterns and performance.
    """
    # Load dataset
    data_path = os.path.join("src", "data", "Outbound_Data.csv")
    df = pd.read_csv(data_path)
    
    # Load data description
    data_description = load_template("Outbound_data.txt")
    
    # Load BI Agent prompt
    bi_prompt = load_template("bi_agent_prompt.txt")
    
    # Define helper functions
    helper_functions = {"execute_analysis": execute_analysis}
    
    # Initialize BI Agent
    agent_instance = BIAgent_Class(
        llm=llm, 
        prompt=bi_prompt, 
        tools=[], 
        data_description=data_description, 
        dataset=df, 
        helper_functions=helper_functions
    )
    
    # Get question using the supervisor chain
    question = get_question(state,supervisor_chain)['direct_response']
    print(f"Question for Insights Agent:\n{question}")
    # Generate response
    response = agent_instance.generate_response(question)
    
    # Store in session state
    if 'bi_agent_responses' not in st.session_state:
        st.session_state.bi_agent_responses = []
    
    bi_response = {
        'question': question,
        'answer': response['answer'],
        'figure': response['figure'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    st.session_state.bi_agent_responses.append(bi_response)
    
    # Display figure if available
    if response['figure']:
        display_saved_plot(response['figure'])
    
    message = response['answer']
    
    return [HumanMessage(content=message)]

# ---------------------- Dynamic Cost Optimization Agent ----------------------

def Dynamic_CostOptimization_Agent(state: AgentState):
    """
    The Dynamic Cost Optimization Agent is responsible for analyzing shipment cost-related data and recommending 
    strategies to reduce or optimize costs. This agent handles tasks such as identifying cost-saving 
    opportunities, calculating the optimal number of trips, performing scenario-based cost optimizations 
    (e.g., varying consolidation windows, truck capacity adjustments), and providing benchmarks and comparisons
    between current and optimized operations. The agent also calculates key performance metrics like cost per 
    pallet, truck utilization rate, and cost savings over time. This agent is called when the user asks about 
    shipment cost reduction or optimization scenarios.
    """

    # Load data
    file_path = os.path.join("src", "data", "Complete Input.xlsx")
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # Get question using the supervisor chain
    question = get_question(state,supervisor_chain)['direct_response']
    print(f"Question for DCO-Agent:\n{question}")
    
    # Set up parameters
    parameters = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "query": question,
        "file_name": file_path,
        "df": df
    }
    
    # Initialize agent and handle query
    agent_instance = AgenticCostOptimizer(llm, parameters)
    response_parameters = agent_instance.handle_query(question)
    
    # Display UI
    show_ui_cost_saving_agent(response_parameters)
    
    # Store in session state
    if 'cost_optimization_response' not in st.session_state:
        st.session_state.cost_optimization_response = []
    
    consolidation_response = {
        'query': question,
        'answer': response_parameters['final_response'].content if hasattr(response_parameters['final_response'], 'content') else response_parameters['final_response'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    st.session_state.cost_optimization_response.append(consolidation_response)
    
    message = consolidation_response['answer']
    
    return [HumanMessage(content=message)]

# ---------------------- Static Cost Optimization Agent ----------------------

def Static_CostOptimization_agent(state: AgentState):
    """
    The Static Cost Optimization Agent is designed to analyze and optimize shipment costs by
    evaluating scenarios before and after consolidation. Using a Rate Card (which includes product type, short postcode, and cost per pallet),
    the agent calculates the base shipment costs. To maximize cost savings, the agent evaluates multiple delivery
    day scenarios (e.g., 5-day, 4-day, or 3-day delivery options).By applying consolidation day mappings, the agent
    aggregates shipments into fewer deliveries, reducing overall costs. The results include: Total shipment costs before and after consolidation ,
    Percentage savings achieved ,Key metrics such as the number of shipments and average pallets per shipment.
    This tool empowers users to identify the most cost-effective delivery strategies while maintaining operational efficiency.
    """

    # Load data
    file_path = os.path.join("src", "data", "Complete Input.xlsx")
    cost_saving_input_df = pd.read_excel(file_path, sheet_name="Sheet1")
    rate_card_path = os.path.join("src", "data", "Cost per pallet.xlsx")
    rate_card = pd.read_excel(rate_card_path)
    
    # Get question using the supervisor chain
    question = get_question(state,supervisor_chain)['direct_response']
    print(f"Question for SCO-Agent:\n{question}")
    
    # Set up parameters
    parameters = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "query": question,
        "complete_input": cost_saving_input_df,
        "rate_card": rate_card
    }
    
    # Initialize agent and handle query
    Static_agent = Static_CostOptimization_Class(llm, parameters)
    response_parameters = Static_agent.handle_query(question)
    
    # Display UI
    show_ui_cost_saving_agent_static(response_parameters)
    
    # Store in session state
    if 'static_optimization_response' not in st.session_state:
        st.session_state.static_optimization_response = []
    
    consolidation_response = {
        'query': question,
        'answer': response_parameters['final_response'].content if hasattr(response_parameters['final_response'], 'content') else response_parameters['final_response'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    st.session_state.static_optimization_response.append(consolidation_response)
    
    message = consolidation_response['answer']
    
    return [HumanMessage(content=message)]

# ---------------------- Generate Scenario Agent ----------------------

def generate_scenario_agent(state: AgentState):
    """
    Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on 
    user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such as 
    the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring different 
    operational scenarios. It can model changes in the system and assess the consequences of those changes to 
    support decision-making and optimization. This agent is called when the user asks about scenario generation,
    comparisons of different outcomes, or analysis of hypothetical situations.
    """

    # Load data
    file_path = os.path.join("src", "data", "Complete Input.xlsx")
    cost_saving_input_df = pd.read_excel(file_path, sheet_name="Sheet1")
    rate_card_path = os.path.join("src", "data", "Cost per pallet.xlsx")
    rate_card = pd.read_excel(rate_card_path)
    
    # Get question using the supervisor chain
    question = get_question(state,supervisor_chain)['direct_response']
    print(f"Question for GS-Agent:\n{question}")
    
    # Ask supervisor to determine which agent to use
    text = """You're inside `generate scenario agent` and your job is to generate a scenario using one of the following agents 
    `Dynamic` or `Static`. Based on the all the given info. Choose one to proceed."""
    
    state['messages'] = add_messages(state['messages'], [HumanMessage(content=text)])
    response = supervisor_chain.invoke(state['messages'])
    state['messages'] = add_messages(state['messages'], [HumanMessage(content=response.get('thought_process', ""))])
    
    message = ""
    
    if response['next'] == 'Dynamic Cost Optimization Agent':
        # Set up parameters
        parameters = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "query": question,
            "file_name": file_path,
            "df": cost_saving_input_df
        }
        
        # Initialize agent and handle query
        agent = AgenticCostOptimizer(llm, parameters)
        response_result = agent.handle_query(question)
        
        # Display UI
        show_ui_cost_saving_agent(response_result)
        
        # Store in session state
        if 'cost_optimization_response' not in st.session_state:
            st.session_state.cost_optimization_response = []
        
        consolidation_response = {
            'query': question,
            'answer': response_result['final_response'].content if hasattr(response_result['final_response'], 'content') else response_result['final_response'],
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.cost_optimization_response.append(consolidation_response)
        
        message = consolidation_response['answer']
        
    elif response['next'] == 'Static Cost Optimization Agent':
        # Set up parameters
        parameters = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "query": question,
            "complete_input": cost_saving_input_df,
            "rate_card": rate_card
        }
        
        # Initialize agent and handle query
        Static_agent = Static_CostOptimization_Class(llm, parameters)
        response_result = Static_agent.handle_query(question)
        
        # Display UI
        show_ui_cost_saving_agent_static(response_result)
        
        # Store in session state
        if 'static_optimization_response' not in st.session_state:
            st.session_state.static_optimization_response = []
        
        consolidation_response = {
            'query': question,
            'answer': response_result['final_response'].content if hasattr(response_result['final_response'], 'content') else response_result['final_response'],
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.static_optimization_response.append(consolidation_response)
        
        message = consolidation_response['answer']
        
    else:
        message = 'Generate Scenario Agent called but no matching scenario type was found.'
    
    return [HumanMessage(content=message)]


# ---------------------- Workflow Setup ----------------------

# Define agent nodes
bi_agent_node = functools.partial(agent_node, agent=bi_agent, name="Insights Agent")
dynamic_cost_optimization_node = functools.partial(agent_node, agent=Dynamic_CostOptimization_Agent, name="Dynamic Cost Optimization Agent")
static_cost_optimization_node = functools.partial(agent_node, agent=Static_CostOptimization_agent, name="Static Cost Optimization Agent")
generate_scenario_agent_node = functools.partial(agent_node, agent=generate_scenario_agent, name="Generate Scenario Agent")


# Define the multi-agent workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Insights Agent", bi_agent_node)
workflow.add_node("Dynamic Cost Optimization Agent", dynamic_cost_optimization_node)
workflow.add_node("Static Cost Optimization Agent", static_cost_optimization_node)
workflow.add_node("Generate Scenario Agent", generate_scenario_agent_node)
workflow.add_node("supervisor", supervisor_node)

# Add edges from agents to supervisor
for member in members:
    if member['agent_name'] != "SELF_RESPONSE":  # Skip self_response as it's not a real node
        workflow.add_edge(member['agent_name'], "supervisor")

# Define conditional routing
conditional_map = {k['agent_name']: k['agent_name'] for k in members if k['agent_name'] != "SELF_RESPONSE"}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Set entry point
workflow.set_entry_point("supervisor")

# Compile the workflow with the memory checkpointer
multi_agent_graph = workflow.compile(checkpointer=memory)