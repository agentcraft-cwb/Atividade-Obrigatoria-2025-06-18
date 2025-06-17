import os
import sys
import types
import uuid
from pathlib import Path
from dotenv import load_dotenv
from operator import add
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from IPython.display import Image, display

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.config import get_stream_writer

from langsmith import Client

from neo4j.exceptions import CypherSyntaxError

from tools.data_ingestion import ingest_files_to_graph_db
from tools.download_and_extract import download_and_extract_zip_from_drive, confirm_if_file_is_on_drive

load_dotenv()

# Setting up LangSmith
LANGSMITH_TRACING=os.getenv('LANGSMITH_TRACING')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT=os.getenv('LANGSMITH_PROJECT')

ls_client = Client(api_key=LANGSMITH_API_KEY)


# Pastas de trabalho
DOWNLOAD_DIR = os.path.join(str(Path.home()), 'Downloads')
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "extracted")

# LLM VAriables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "gpt-4.1-nano"
MODEL_CYPHER = "gpt-4o"

#Graph variables and connection setps
DB_NAME = "neo4j"
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,
        )
except:
    print("Please make sure a Neo4j db is running wiyh the APOCs pluggin installed.")
    sys.exit()

MAX_TRIALS = 3 # Max trials to correct cypher statement

# defining the Output and Overall state of the LangGraph application

class OverallState(TypedDict):
    status_stream: bool #Para fazer ou não o broadcasting to status
    hitl_prompt: str #Message to interrupt
    file_id: str # The id of the file to download
    graph: str # After download and execute and ingest, the graph driver
    question: str #Where we'll be storing the question for the query construction
    database_records: List[dict] #DB records returned from cypher statement
    ignore_msgs: int #Index of human messages to ignore for question seartch
    next_action: str #To mage conditional edges
    cypher_statement: str #Latetest cypher statement to be used
    cypher_errors: List[str] #List of errors when verifying cypher statemen
    cypher_history: Annotated[List[str], add] # To track the history of cypher statement correction
    trials_count: int # To track how many retrials to generate good cypher statements (to limit retrials)
    token_count: Annotated[int, add] # from LLM invocations
    cost_count: Annotated[float, add] # from LLM invocations
    steps: Annotated[List[str], add] # Tracking of agentic steps through the process
    messages: Annotated[list[AnyMessage], add_messages] # History of messages exchanged
    
# Little wrapping function to help on counting tokens and cost

def invoke_and_count(llm_call, inputs):
    with get_openai_callback() as cb:
        output = llm_call.invoke(inputs)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost
    
    return output, total_tokens, total_cost

###########################################################################
#                               NODES
###########################################################################

# #####################################
# FIRST NODE: Verify File to download
#######################################

def verify_file(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    Verify the zip file name by parsing it's name and searching the google drive folder.
    
    This node handles file name verification as the first step in the conversation.
    It extracts zip file name to be downloaded from user messages and validates
    them against the google drive folder.
    
    Args:
        state (State): Current state containing messages and potentially file name
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with file_id if verified, or request for more info
    """

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    class FileToDownload(BaseModel):
        """Schema for parsing zip file information."""
        zip_file: str = Field(description="Name of the zip file to be searched in the folder")

    # System prompt for extracting customer identifier information
    structured_system_prompt = """You are chatbot responsible for identifying the name of the zip 
    file the user wants to download from the i2a2 google drive.
    Only extract the name of the zip file from the message history. 
    If they haven't provided the information yet, return an empty string for the identifier."""

    guardrails_prompt = ChatPromptTemplate.from_messages(
        [("system", structured_system_prompt,),
        ("human",("{message}"),),])

    # Create a structured LLM that outputs responses conforming to the UserInput schema
    get_file_name_chain = guardrails_prompt|llm.with_structured_output(FileToDownload)

    # setup status stream writer
    writer = get_stream_writer()
    
    # Get the most recent user message
    user_input = state['messages']

    # Use structured LLM to parse customer identifier from the message
    if state['status_stream']:
        writer({"custom_key": {"node": "verify_file", "status": "Verifying file to download"}})
    parsed_info, tokens, cost = invoke_and_count(get_file_name_chain, {"message": [user_input][-1]})
        
    # Extract the identifier from parsed response
    zip_file = parsed_info.zip_file

    # Attempt to find the customer ID using the provided identifier
    if (zip_file):
        if state['status_stream']:
            writer({"custom_key": {"node": "verify_file", "status": "Verifying file existance in drive"}})
        file = confirm_if_file_is_on_drive(zip_file)

        # If file found, confirm verification and set customer_id in state
        if file:
            intent_message = AIMessage(
                content= f"Obrigado por me passar o nome do arquivo. Consegui confirmar a identificação do arquivo {file['file_name']}. Vou prosseguir com o download, extração e ingestão."
            )
            return {
                "file_id": file['file_id'],
                "messages" : [intent_message],
                "steps": ["verify_file:OK"],
                "cost_count": cost,
                "token_count": tokens,
                "next_action": "dload_n_xtract"
                }
        else:
            intent_message = AIMessage(
                content= f"Obrigado. Tentei encontrar o arquivo {zip_file}, mas não consegui. Você pode, por favo verificar e confirmar nome correto?"
            )
            return {
                "messages": [intent_message], 
                "steps": ["verify_file:NOK"], 
                "cost_count": cost, 
                "token_count": tokens,
                "next_action": "human_input",
                "hitl_prompt": "Inform file name"
                }
    else:
        # If file not found, ask for correct information
        if state['status_stream']:
            writer({"custom_key": {"node": "verify_file", "status": "Identifying file name"}})
        system_prompt = ls_client.pull_prompt("atividade1806-verify_file_01")
        prompt = ChatPromptTemplate.from_messages([system_prompt])
        messages = state.get("messages")
        if len(messages) > 4 :
            messages = messages[-4]
        llm = ChatOpenAI(model=MODEL_NAME, temperature=1)
        chain_llm = prompt|llm
        response, tokens2, cost2 = invoke_and_count(chain_llm, messages)
        return {
            "messages": [response], 
            "steps": ["verify_file:NOK"], 
            "cost_count": cost + cost2, 
            "token_count": tokens + tokens2,
            "next_action": "human_input",
            "hitl_prompt": "Inform file name"
            }


#########################################################
# SECOND NODE: HUMAN IN THE LOOP 1 - jUST TO GET FILENAME
#########################################################

def human_input(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    Human-in-the-loop node that interrupts the workflow to request user input.
    
    This node creates an interruption point in the workflow, allowing the system
    to pause and wait for human input before continuing. It's typically used
    for customer verification or when additional information is needed.
    
    Args:
        state (State): Current state containing messages and workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the user's input message
    """
    prompt = state.get("hitl_prompt")
    # Interrupt the workflow and prompt for user input
    interrupt(prompt)

###########################################
# THIRD NODE Download and Extract node
#########################################


def dload_n_xtract(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    Downloads and extracts the file id (state) from the google drive folder id
    '1EYgJrhf3BKHypPQLT5xwTHhsHa2BYMFt' to the default 'Downloads" folder.
    """
    writer = get_stream_writer()

    file_id = state.get("file_id")

    if state['status_stream']:
        writer({"custom_key": {"node": "dload_n_xtrack", "status": "Downloading zip file"}})
    result = download_and_extract_zip_from_drive(
        file_id,
        DOWNLOAD_DIR,
        EXTRACT_DIR
    )

    if "sucess" in result:
        if state['status_stream']:
            writer({"custom_key": {"node": "dload_n_xtrack", "status": f"Successfull download and extraction of files {result['sucess']} ."}})
        return {
            "message": [AIMessage(content=f"Arquivo ZIP baixado com sucesso e arquivos {result['sucess']} extraídos com sucesso.")],
            "steps": ["dload_n_xtract"],
            "next_action": "data_ingestion"
        }
    else:
        if state['status_stream']:
            writer({"custom_key": {"node": "dload_n_xtrack", "status": result['error']}})
        return { "message": [AIMessage(content=result['error'])],
                "steps": ["dload_n_xtract:NOK"],
                "next_action": "end",
                "database_records": result
                }

####################################
# NODE FOUR - DATA INGESTION
###################################

def data_ingestion(state: OverallState, config: RunnableConfig) -> OverallState:
    '''
    Takes the two invoice files (Heads and Items) and populate the neo4j graph database
    while alse creating the several indexes for later effective queries
    '''
    result = ingest_files_to_graph_db()

    return {
        "messages": [AIMessage(content="CSV files ingested")],
        "steps": ["data_ingestion"],
        "next_action": "get_inquiry",
        "database_records": result,
        "graph": "checked"
    }

####################################
# NODE FIVE - GET INQUIRY
###################################

def get_inquiry(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    Verify the user messages to fint the inquiry to the databas
    
    Args:
        state (State): Current state containing messages and potentially file name
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with question, or request for more info
    """

    class UserInquiry(BaseModel):
        """Schema for parsing zip file information."""
        question: str = Field(description="The inquiry the user wants to make to the invoice documents ingested")

    # System prompt for extracting a question from the previous messages
    system_prompt = ls_client.pull_prompt("atividade1806-get_inquiry-01")

    inquiry_prompt = ChatPromptTemplate.from_messages([
        (system_prompt),
        ( "human", ("{human_messages}"),),
        ])

    # Creates a structured LLM that identifies the question the user wants to make
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    get_inquiry_chain = inquiry_prompt|llm.with_structured_output(UserInquiry)
    
    # System prompt for extracting customer identifier information
    pose_a_question_system_prompt = ls_client.pull_prompt("atividade1806-get_inquiry-02")

    pose_a_question = ChatPromptTemplate.from_messages([(pose_a_question_system_prompt)])
    inquiry_not_found_chain = pose_a_question|llm

    writer = get_stream_writer()

    # Get user messages
    human_messages = [msg.content for msg in state.get("messages") if msg.type == "human"]
    
    # Use structured LLM to parse customer identifier from the message
    if state['status_stream']:
        writer({"custom_key": {"node": "get_inquiry", "status": "identifying question"}})
    parsed_info, token_count, cost_count = invoke_and_count(get_inquiry_chain, {"human_messages": human_messages[-1]})
        
    # Extract the identifier from parsed response
    question = parsed_info.question

    # Attempt to find the customer ID using the provided identifier
    if (question):
        return {
                "question": question,
                "steps": ["get_inquiry:OK"],
                "cost_count": cost_count,
                "token_count": token_count,
                "ignore_msgs": len(human_messages),
                "next_action": "guardrails"
                }
    else:
        # If question not clear or found, ask user to rephrase
        ai_messages = [msg.content for msg in state.get("messages") if msg.type == "ai"]
        if state['status_stream']:
            writer({"custom_key": {"node": "get_inquiry", "status": "Failed to identify question, waiting for user prompt"}})
        response, token_count2, cost_count2 = invoke_and_count(inquiry_not_found_chain, {"messages": ai_messages, "schema": Schema})
        return {
            "question": None,
            "messages": [response], 
            "steps": ["get_inquire:NOK"], 
            "cost_count": cost_count + cost_count2, 
            "token_count": token_count + token_count2,
            "next_action": "human_input",
            "hitl_prompt": "Please folmulate question"
            }

####################################################
# NODE SEVEN - GUARDRAIL
##################################################

def guardrails(state: OverallState) -> OverallState:
    """
    Decides if the question is related to invoices or not.
    """

    guardrails_system = """
    As an intelligent assistant, your primary objective is to decide whether a given question is related to the invoice dataset or not. 
    If the question is related to the invoices, output "invoice". Otherwise, output "end".
    To make this decision, assess the content of the question and determine if it refers to invoices emitted or issued to companies, 
    which state (uf) these companies are locatede, the contents or invoice items of each invoice and the products and services they 
    relate to. Provide only the specified output: "invoice" or "end".
    """
    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                guardrails_system,
            ),
            (
                "human",
                ("{question}"),
            ),
        ]
    )

    class GuardrailsOutput(BaseModel):
        decision: Literal["go", "no_go"] = Field(
            description="Decision on whether the question is related to the invoices"
        )

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

    writer = get_stream_writer()

    if state['status_stream']:
        writer({"custom_key": {"node": "guardrail", "status": "Testing guardrails"}})
    guardrails_output, tokens, cost = invoke_and_count(guardrails_chain, {"question": state.get("question")})
    
    database_records = None

    if guardrails_output.decision == "end":
        database_records = "A pergunta não está relacionada às notes fiscais e seus conteúdos. Não poderei lhe responder."
    return {
        "next_action": guardrails_output.decision,
        "database_records": database_records,
        "steps": ["guardrail"],
        "trials_count": 0,
        "token_count": tokens,
        "cost_count": cost
    }

########################
# Cypher generation chain
#########################

# FEW SHOT PROPMPTING DEFINITION FOR CYPHER GENERATION

#####################################
# NODE EIGHT - CYPHER GENERATION
#####################################

def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input
    """

    examples = [
            {
                "question": "Quais as a empresa que são citadas nos arquivos?",
                "query": "MATCH (p:party) RETURN DISTINCT p.name AS Nome ORDER BY Nome",
            },
            {
                "question": "Quais as empresas que realizaram mais compras no período?",
                "query": "MATCH (p:party)<-[:ISSUED_TO]-(:invoice) WITH p, count(*) AS Numero_de_pedidos RETURN p.name AS Empresa, Numero_de_pedidos ORDER BY Numero_de_pedidos DESC LIMIT 5",
            },
            {
                "question": "Quais as empresas que mais emitiram notas entre os dias 5 de Janeiro de 2024 e o dia 10 de Janeiro de 2024?",
                "query": "MATCH (p:party)<-[:ISSUED_BY]-(i:invoice) WHERE i.issueDateTime >= datetime('2024-01-05') AND i.issueDateTime <= datetime('2024-01-20') WITH p, count(i) AS Numero_de_notas WHERE Numero_de_notas > 1 RETURN p.name AS Empresa, Numero_de_notas ORDER BY Numero_de_notas DESC, Empresa LIMIT 5",
            },
            {
                "question": "Quais os produtos que tiveram mais vendas, em quantidade de produtos vendidos?",
                "query": "MATCH (ps:productService)<-[:REFERS_TO_PRODUCT]-(ii:invoiceItem) WITH ps, sum(ii.quantity) AS Total_Vendido WHERE Total_Vendido > 1 RETURN ps.description AS Descrição_do_produto, Total_Vendido ORDER BY Total_Vendido DESC LIMIT 5",
            },
            {
                "question": "Quais os produtos que mais vendidos, em valores?",
                "query": "MATCH (ps:productService)<-[:REFERS_TO_PRODUCT]-(ii:invoiceItem) WITH ps, sum(ii.totalItemPrice) AS Total_valor_vendido WHERE Total_valor_vendido > 1 RETURN ps.description AS Descrição_do_produto, Total_valor_vendido ORDER BY Total_valor_vendido DESC LIMIT 5",
            },
            {
                "question": "Qual a empresa que mais faturou no entre os dias 1 de Janeiro de 2024 e 15 de Janeiro de 2024?",
                "query": "MATCH (p:party)<-[:ISSUED_BY]-(i:invoice) WHERE i.issueDateTime >= datetime('2024-01-20') AND i.issueDateTime <= datetime('2024-01-30') WITH p, sum(i.totalValue) AS Total_faturado RETURN p.name AS Empresa, Total_faturado ORDER BY Total_faturado DESC LIMIT 1",
            },
            {
                "question": "Quais foram as empresas que compraram comida?",
                "query": f"WITH genai.vector.encode('produtos alimentícios', 'OpenAI', {{token:'{OPENAI_API_KEY}', model:'text-embedding-3-large'}}) AS query_vector CALL db.index.vector.queryNodes('productServiceDescriptions_vectIdx', 600, query_vector) YIELD node, score WHERE score > 0.65 MATCH (node)<-[:REFERS_TO_PRODUCT]- (ii:invoiceItem)<-[:HAS_ITEM]-(i:invoice)-[:ISSUED_TO]->(p:party) RETURN p.name AS Nome_da_empresa, node.description AS Descrição, node.ncmshDescription AS DescriçãoNCMSH, score ORDER BY  score DESC",
            },
            {
                "question": "Alguma empresa comprou chocolate?",
                "query": f"WITH genai.vector.encode('chocolate', 'OpenAI', {{token:'{OPENAI_API_KEY}', model:'text-embedding-3-large'}}) AS query_vector CALL db.index.vector.queryNodes('productServiceDescriptions_vectIdx', 600, query_vector) YIELD node, score WHERE score > 0.65 MATCH (node)<-[:REFERS_TO_PRODUCT]- (ii:invoiceItem)<-[:HAS_ITEM]-(i:invoice)-[:ISSUED_TO]->(p:party) RETURN DISTINCT p.name AS Nome_da_empresa, node.description AS Descrição, score",
            },
            {
                "question": "Qual o produto mais caro vendido por uma empresa do RJ?",
                "query": "MATCH (ps:productService)<-[:REFERS_TO_PRODUCT]-(ii:invoiceItem)<-[:HAS_ITEM]-(:invoice)-[:ISSUED_BY]->(:party)-[:LOCATED_IN]->(:state {uf: 'RJ'}) RETURN ii.unitPrice AS Preco_do_produto, ps.description ORDER BY Preco_do_produto DESC LIMIT 1",
            },
            {
                "question": "Temos algum exemplo de 'operação triangular', ou operação 'remessa e retorno'?",
                "query": "MATCH (p1:party)<-[:ISSUED_BY]-(:invoice)-[:ISSUED_TO]->(p2:party)<-[:ISSUED_BY]-(i:invoice)-[:ISSUED_TO]->(p3:party) WHERE p1.identifier=p3.identifier RETURN p1.name, p2.name, i.chave"
            },
            {
                "question": "Temos algum exemplo de operação 'vai e volta' envolvendo três empresas, ou 'Industrialização por Conta' ou 'Ordem de Terceiro'?",
                "query": "MATCH (p1:party)<-[:ISSUED_BY]-(:invoice)-[:ISSUED_TO]->(p2:party)<-[:ISSUED_BY]-(i1:invoice)-[:ISSUED_TO]->(p3:party)<-[:ISSUED_BY]-(i2:invoice)-[:ISSUED_TO]->(p4:party) WHERE p1.identifier=p4.identifier RETURN p1.name, p2.name, p3.name, i1.chave, i2.chave"
            }
        ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, OpenAIEmbeddings(), Neo4jVector, k=5, input_keys=["question"]
        )

    text2cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Given an input question, convert it to a Cypher query. No pre-amble."
                    "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                ),
            ),
            (
                "human",
                (
                    """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
    Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
    Here is the schema information
    {schema}

    Below are a number of examples of questions and their corresponding Cypher queries.

    {fewshot_examples}

    User input: {question}
    Cypher query:"""
                ),
            ),
        ]
    )
    
    graph.refresh_schema()
    writer = get_stream_writer()
    NL = "\n"
    fewshot_examples = (NL * 2).join(
        [
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )

    llm = ChatOpenAI(model = MODEL_CYPHER, temperature = state.get("trials_count")/5  ) #reset temperature in case re-creation is needed
    text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

    if state['status_stream']:
        writer({"custom_key": {"node": "generate_cypher", "status": "generating cypher query"}})
    generated_cypher, total_tokens, total_cost = invoke_and_count(
        text2cypher_chain, 
        {"question": state.get("question"), "fewshot_examples": fewshot_examples, "schema": graph.schema,})
 
    trials = state.get("trials_count") + 1

    return {
        "cypher_statement": generated_cypher, 
        "cypher_history": [f"generation {trials-1}:" + generated_cypher],
        "trials_count": trials,
        "steps": ["generate_cypher"],
        "token_count": total_tokens,
        "cost_count": total_cost,
        "next_action": "validate_cypher"
        }

################################################
# NODE NINE - VALIDATE QUERY STATEMENT
###############################################

def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    

    # Cypher query corrector corrects relatnship directions deterministically
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in graph.structured_schema.get("relationships")
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    validate_cypher_system = """
    You are a Cypher expert reviewing a statement written by a junior developer.
    """

    validate_cypher_user = """You must check the following:
    * Are there any syntax errors in the Cypher statement?
    * Are there any missing or undefined variables in the Cypher statement?
    * Are any node labels used in the cypher statement that are not listed in the schema?
    * Are any relationship types used in the cypher statement that are not listed in the schema?
    * Ae all the relationshio directions used in the cypher statement (-[]-> or <-[]-) in conformity with the schema?
    * Are any of the properties used in the cypher statement not included in the schema?
    * Does the Cypher statement include enough information to answer the question?

    Examples of good errors:
    * Label (:product) does not exist, did you mean (:productService)?
    * Property Item_Number does not exist for label invoiceItem, did you mean itemNumber?
    * Relationship [hasItem] does not exist, did you mean [HAS_ITEM]?
    * Relationship [:ISSUED_BY] runs from (:invoice) to (:party), as (:invoice)-[:ISSUED_BY]->(:party), not the other way around.

    Schema:
    {schema}
    {list_of_index}

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    Make sure you don't make any mistakes!"""

    validate_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                validate_cypher_system,
            ),
            (
                "human",
                (validate_cypher_user),
            ),
        ]
    )

    class Property(BaseModel):
        """
        Represents a filter condition based on a specific node property in a graph in a Cypher statement.
        """
        node_label: str = Field(
            description="The label of the node to which this property belongs."
        )
        property_key: str = Field(description="The key of the property being filtered.")
        property_value: str = Field(
            description="The value that the property is being matched against. Do not include finding on 'RETURN' statements, only those on 'WHERE' statements."
        )

    class ValidateCypherOutput(BaseModel):
        """
        Represents the validation result of a Cypher query's output,
        including any errors and applied filters.
        """
        errors: Optional[List[str]] = Field(
            description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
        )
        filters: Optional[List[Property]] = Field(
            description="A list of property-based filters applied in the Cypher statement."
        )

    llm = ChatOpenAI(model=MODEL_CYPHER, temperature=0)
    validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)

    writer = get_stream_writer()
    errors = []
    mapping_errors = []
    relationship_corrections = 0
    # Check for syntax errors
    try:
        graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message) # In case os Sintax error, list the in errors

    # Experimental feature for correcting relationship directions
    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        relationship_corrections = relationship_corrections + 1

    # Use LLM to find additional potential errors and get the mapping for values
    if state['status_stream']:
        writer({"custom_key": {"node": "validate_cypher", "status": "checking for cypher statement errors"}})

    list_of_index = get_indexes(graph)

    llm_output, total_tokens, total_cost = invoke_and_count(
        validate_cypher_chain,
        {"question": state.get("question"),"schema": graph.schema,"list_of_index": list_of_index,"cypher": state.get("cypher_statement"),}
        )

    if llm_output.errors:
        errors.extend(llm_output.errors)
    #if llm_output.filters:
    #    for filter in llm_output.filters:
    ##        # Do mapping only for string values
    #        if (
    #            not [
    #                prop
    #                for prop in graph.structured_schema["node_props"][
    #                    filter.node_label
    #                ]
    #                if prop["property"] == filter.property_key
    #####            ][0]["type"]
        #        == "STRING"
    #            ):
    #            continue
    #        mapping = graph.query(
    #            f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
    #            {"value": filter.property_value},
    #            )
    #        if not mapping:
    #            print(f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}")
    #            mapping_errors.append(
    ##                f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
    #                )
    if mapping_errors:
        next_action = "correct_cypher"
    elif errors:
        if state.get('trials_count') < MAX_TRIALS:
            next_action = "generate_cypher"
        else:
            next_action = "manual_query"
    else:
        next_action = "execute_cypher"

    if state.get('cypher_statement')==corrected_cypher:
        return {
            "next_action": next_action,
            "cypher_statement": corrected_cypher,
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
            "token_count": total_tokens,
            "cost_count": total_cost
        }
    else:
        return {
            "next_action": next_action,
            "cypher_statement": corrected_cypher,
            "cypher_history": [f"corrected x{relationship_corrections}:" + corrected_cypher],
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
            "token_count": total_tokens,
            "cost_count": total_cost
        }
    
########################################
# NODE TEN - CORRECTIONG CYPHER ERRORS
#########################################

correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}
{list_of_index}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

llm = ChatOpenAI(model=MODEL_CYPHER, temperature=0)
correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()

def get_indexes(graph) -> str:
    indexes = graph.query("SHOW INDEXES WHERE type = 'VECTOR'")
    list_of_index = "The vector embedding indexes:\n"
    for index in indexes:
        list_of_index = list_of_index + f"-name: {index['name']},\n-labelsOrTypes: {index['labelsOrTypes']}\n"
    return list_of_index

def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    
    list_of_index = get_indexes(graph)
    writer = get_stream_writer()
    if state['status_stream']:
        writer({"custom_key": {"node": "correct_cypher", "status": "correcting cypher statement"}})
    corrected_cypher, total_tokens, total_cost = invoke_and_count(
        correct_cypher_chain,
        {
        "question": state.get("question"),
        "errors": state.get("cypher_errors"),
        "cypher": state.get("cypher_statement"),
        "schema": graph.schema,
        "list_of_index": list_of_index
        })

    return {
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "cypher_history": ["corrected:" + corrected_cypher],
        "steps": ["correct_cypher"],
        "token_count": total_tokens,
        "cost_count": total_cost
    }

###############################################
# NODE ELEVEN - TEST CYPHER STATEMENT
###############################################


def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """
    no_results = "Nenhuma informação relevante foi encontrada na base de dados"
    writer = get_stream_writer()
    if state['status_stream']:
        writer({"custom_key": {"node": "execute_cypher", "status": "Excetute query"}})
    try:
        records = graph.query(state.get("cypher_statement"))
        next_action = "generate_final_answer"
        records = records if records else no_results
    except:
        next_action = "generate_cypher"
        records = "No records, cypher failed"

    return {
        "database_records": records,
        "next_action": next_action,
        "steps": ["execute_cypher"]
    }

###############################################
# NODE TWELVE - GENERATING FINAL ANSWER
##############################################

generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a complete but succinct, definitive answer to the user's question. Take into account that the results may contain multiple lines. Take 
all lines into consideration.

Respond as if you are answering the question directly. If you have no question, just explain the results.

Results: {results}
Question: {question}

After answering the question, please ask the user if he has any other question, if he want's to work on a 
diferent file, or if he wants to exit.

Remember, you are brazilian and only speaks in Portuguese."""
            ),
        ),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
generate_final_chain = generate_final_prompt | llm | StrOutputParser()

def generate_final_answer(state: OverallState) -> OverallState:
    """
    Takes query results (or no result) depending of flow, and elaborate final answer.
    """
    writer = get_stream_writer()
    if state['status_stream']:
        writer({"custom_key": {"node": "generate_final_answer", "status": "Generating Final answer"}})
    final_answer, token_count, cost_count = invoke_and_count(generate_final_chain, {"question": state.get("question"), "results": state.get("database_records")})

    return {
        "messages": [final_answer], 
        "steps": ["generate_final_answer"], 
        "cost_count": cost_count,
        "token_count": token_count,
        "next_step": "human_input",
        "hitl_prompt": "Please tell me what to do next"
        }
###############################################################################################
# MANUAL QUERY
##############################################################################################

def manual_query(state:OverallState) -> OverallState:
    last_cypher = state.get('cypher_statement')
    manual_cypher = input(f"AGENT: I have last tried with this cypher: {last_cypher}. Please try with your own!")
    return {
        "next_action": "execute_cypher",
        "cypher_statement": manual_cypher,
        "cypher_history": ["manual_entry:" + manual_cypher],
        "steps": ["manual_query"],
        "trials_count": 0
    }

#####################################################################################################
# FLOW MANAGER
#
# The flowmanager's role is to distribute the task to other nodes depending on the state of the agent
# Some of the decision can be deterministic. The HITL means that needs input from user before going to 
# next node.
# - if state.get("file_id") is None -HITL-> verity_file
# - elif state.get("graph") is None -> dload_n_xtract
# - elif state.get("question") is None -HITL-> get_inquiry
#
# 
####################################################################################################

def flowmanager(state:OverallState) -> OverallState:
    
    file_id = state.get("file_id")
    graph = state.get("graph")
    question = state.get("question")

    if file_id is None:
        return {
            "next_action": "verify_file",
            "steps": ["flowmanager"]
            }
    elif graph is None:
        return {
            "next_action": "dload_n_xtract",
            "steps": ["flowmanager"]
            }
    elif question is None:
        return {
            "next_action": "get_inquiry",
            "steps": ["flowmanager"]
            }
    else:
        # Once the question is answered, tetemine if user wants to
        # - Ask another question
        # - Query db manually
        # - Get an other ZIP file
        # - exit
        system_prompt = """
            You are a helpfull assistant. Your job at this point is to determine what the user wants to do next.
            His options are:
            - Ask a quastion to the databas. Note that he might not say "I want ro ask a question", he will simply pose a question).
            - Query the datapabes manually with a manual cypher query. If whe wants to do so he must be explicit about it. 
            - Restart the full process by downloading a new file. Again, in this case he will either be very explicit abput it or just give you a .zip file name.
            - Exit the agent workflow. Again, user must be explicit about it.
            """
        
        class FlowmanagerDecision(BaseModel):
            decision: Literal["ask question", "manual cypher query", "new file", "exit"] = Field(
                description="Decision on what next step to take"
            )
    
        flowmanager_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt,),
            ("human", ("{message}"),),])
        
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        flowmanager_chain = flowmanager_prompt|llm.with_structured_output(FlowmanagerDecision)
        message = state.get("messages")[-1]
  
        writer = get_stream_writer()
        if state['status_stream']:
            writer({"custom_key": {"node": "flow_manager", "status": "Thinking"}})

        response, token, cost = invoke_and_count(flowmanager_chain, {"message": message} )
        
        if response.decision == "ask question":
            return {
                "steps": ["flow_manager"],
                "next_action": "get_inquiry",
                "database_records": None,
                "question": None,
                "token_count": token,
                "cost_count": cost
            }
        elif response.decision == "new file":
            return {
                "steps": ["flow_manager"],
                "next_action": "verify_file",
                "database_records": None,
                "question": None,
                "file_id": None,
                "token_count": token,
                "cost_count": cost
            }
        elif response.decision == "manual cypher query":
            return {
                "steps": ["flow_manager"],
                "next_action": "manual_query",
                "database_records": None,
                "token_count": token,
                "cost_count": cost
            }
        else:
            return {
                "steps": ["flow_manager"],
                "next_action": "exit",
            }
 
def finishit(state:OverallState):
    writer = get_stream_writer()
    if state['status_stream']:
        writer({"custom_key": {"node": "finishit", "status": "Exiting graph"}})
    return {"messages": [AIMessage("Eu que agradeço. Volte sempre!")]}

######################################################################################################
# GRAPH CONSTRUCTION
#######################################################################################################

def flowmanager_conditional(state: OverallState, config: RunnableConfig)\
    -> Literal["verify_file", "dload_n_xtract", "get_inquiry", "manual_query", "exit"]:
    next_action = state.get("next_action")
    if next_action == "verify_file":
        return "verify_file"
    elif next_action == "dload_n_xtract":
        return "dload_n_xtract"
    elif next_action == "get_inquiry":
        return "get_inquiry"
    elif next_action == "manual_query":
        return "manual_query"
    elif next_action == "exit":
        return "finishit" 

def verify_file_conditional(state: OverallState, config: RunnableConfig) -> Literal["flowmanager", "human_input"]:
    if state.get("file_id") is not None:
        return "flowmanager" # Customer ID is verified, continue to the next step (supervisor)
    else:
        return "human_input" # Customer ID is not verified, interrupt for human input
    
def get_inquiry_conditional(state: OverallState, config: RunnableConfig) -> Literal["guardrails", "human_input"]:
    if state.get("question") is not None:
        return "guardrails" # have the question, continue to the next step (guardrail)
    else:
        return "human_input" # Customer ID is not verified, interrupt for human input
    
def guardrails_conditional(state: OverallState, config: RunnableConfig) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "no_go":
        return "generate_final_answer"
    elif state.get("next_action") == "go":
        return "generate_cypher"

def validate_cypher_conditional(state: OverallState) -> \
    Literal["manual_query", "execute_cypher", "correct_cypher", "generate_cypher"]:
    if state.get("next_action") == "manual_query":
        return "manual_query"
    elif state.get("next_action") == "execute_cypher": 
        return "execute_cypher"
    elif state.get("next_action") == "correct_cypher": 
        return "correct_cypher"
    elif state.get("next_action") == "generate_cypher": 
        return "generate_cypher"
    
def execute_cypher_conditional(state: OverallState)\
    -> Literal["generate_cypher", "generate_final_answer"]:

    if state.get("next_action") == "generate_final_answer":
        return "generate_final_answer"
    elif state.get("next_action") == "generate_cypher":
        return "generate_cypher"

langgraph = StateGraph(OverallState)

langgraph.add_node(flowmanager)
langgraph.add_node(verify_file)
langgraph.add_node(human_input)
langgraph.add_node(dload_n_xtract)
langgraph.add_node(data_ingestion)
langgraph.add_node(get_inquiry)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(manual_query)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)
langgraph.add_node(finishit)

langgraph.add_edge(START, "flowmanager")
langgraph.add_conditional_edges("flowmanager", flowmanager_conditional,
                                    {
        "verify_file": "verify_file", 
        "dload_n_xtract": "dload_n_xtract",
        "get_inquiry": "get_inquiry",
        "manual_query": "manual_query",
        "finishit": "finishit"
    })
langgraph.add_conditional_edges("verify_file", verify_file_conditional)
langgraph.add_edge("human_input", "flowmanager")
langgraph.add_edge("dload_n_xtract", "data_ingestion") 
langgraph.add_edge("data_ingestion","flowmanager")
langgraph.add_conditional_edges("get_inquiry", get_inquiry_conditional)
langgraph.add_conditional_edges("guardrails", guardrails_conditional)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges("validate_cypher", validate_cypher_conditional,
                                    {
        "manual_query": "manual_query",
        "execute_cypher":"execute_cypher",
        "generate_cypher": "generate_cypher",
        "correct_cypher": "correct_cypher"
    })
langgraph.add_edge("manual_query", "execute_cypher")
langgraph.add_conditional_edges("execute_cypher", execute_cypher_conditional)
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", "human_input")
langgraph.add_edge("finishit", END)

checkpointer = MemorySaver() #To allow continuity of conversation after interuption

app = langgraph.compile(checkpointer=checkpointer)

async def call(self, question: str, jump_state=False, status_stream=True):
    if self.config:
        initial_state = {
            "status_stream": status_stream,
            "messages": HumanMessage(question)}
    else:
        thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": thread_id}}
        if jump_state:
            initial_state = {
                "status_stream": status_stream,
                "ignore_msgs": 0,
                "token_count": 0,
                "cost_count": 0,
                "file_id": "somefile",
                "graph": "checked",
                "question": question,
                "messages": HumanMessage(question)
            }
        else:
            initial_state = {
                "status_stream": status_stream,
                "ignore_msgs": 0,
                "token_count": 0,
                "cost_count": 0,
                "messages": HumanMessage(question)}
    
    async for event in self.astream_events(initial_state, config=self.config, version="v2", stream_mode="custom"):
        if "chunk" in event['data']: 
            if event["data"]["chunk"] is not None:
                if "custom_key" in event["data"]["chunk"]:
                    print(event["data"]["chunk"]["custom_key"])
    for message in event['data']['output']['messages']:
        message.pretty_print()
    return app.get_state(self.config)

def view_graph(self):
    return Image(self.get_graph().draw_mermaid_png())

def turbinar_app(app):
    app.view = types.MethodType(view_graph, app)
    app.call = types.MethodType(call, app)
    app.config = None
    return app

def compilar_agente():
    turbo_app = turbinar_app(app)
    return turbo_app