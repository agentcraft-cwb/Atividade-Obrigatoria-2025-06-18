from neo4j import GraphDatabase
import os
import pandas as pd

from langgraph.config import get_stream_writer

NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

URI = "bolt://localhost:7687"
DB_NAME = "neo4j"
AUTH = (DB_NAME, NEO4J_PASSWORD)

HEADER_FILE = "C:/Users/nikin/Downloads/extracted/202401_NFs_Cabecalho.csv"
ITEM_FILE = "C:/Users/nikin/Downloads/extracted/202401_NFs_Itens.csv"

Estados = {
    "AC":"Acre",
    "AL":"Alagoas",
    "AP":"Amapá",
    "AM":"Amazonas",
    "BA":"Bahia",
    "CE":"Ceará",
    "DF":"Distrito Federal",
    "ES":"Espírito Santo",
    "GO":"Goiás",
    "MA":"Maranhão",
    "MT":"Mato Grosso",
    "MS":"Mato Grosso do Sul",
    "MG":"Minas Gerais",
    "PA":"Pará",
    "PB":"Paraíba",
    "PR":"Paraná",
    "PE":"Pernambuco",
    "PI":"Piauí",
    "RJ":"Rio de Janeiro",
    "RN":"Rio Grande do Norte",
    "RS":"Rio Grande do Sul",
    "RO":"Rondônia",
    "RR":"Roraima",
    "SC":"Santa Catarina",
    "SP":"São Paulo",
    "SE":"Sergipe",
    "TO":"Tocantins",
    }

def import_batch(driver, nodes, batch_n):
    # Generate and store embeddings
    driver.execute_query('''
    CALL genai.vector.encodeBatch($listToEncode, 'OpenAI', { token: $token, model:"text-embedding-3-large", dimentions:3072}) YIELD index, vector
    MATCH (ps:productService {description: $ProductServices[index].description})
    CALL db.create.setNodeVectorProperty(ps, 'embedding', vector)
    ''', ProductServices=nodes, listToEncode=[ProductService['to_encode'] for ProductService in nodes], token=OPENAI_API_KEY,
    database_=DB_NAME)

def ingest_files_to_graph_db(heads = HEADER_FILE, items = ITEM_FILE):
    '''
    Takes the two invoice files (Heads and Items) and populate the neo4j graph database
    while alse creating the several indexes for later effective queries
    '''
    writer = get_stream_writer()

    # open the files into pandas dfs
    df_heads = pd.read_csv(heads)
    df_items = pd.read_csv(items)

    # Effectively connects to graph and create driver
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

    # Loop through Header file and create invoice, party and state nodes and it´s connections
    nodes_count = 0
    time_count = 0
    relationships_count = 0
    labels_count = 0
    properties_count = 0

    for index, row in df_heads.iterrows():

        cypher = f'''
            MERGE (i:invoice {{ number: {str(row['NÚMERO'])} }})
            ON CREATE SET i.number = {str(row['NÚMERO'])},
            i.chave = "{row['CHAVE DE ACESSO']}",
            i.model = "{row['MODELO']}",
            i.series = {str(row['SÉRIE'])},
            i.operatinNature = "{row['NATUREZA DA OPERAÇÃO']}",
            i.issueDateTime = datetime("{str(row['DATA EMISSÃO']).replace(' ', 'T')}"),
            i.totalValue = {str(row['VALOR NOTA FISCAL'])},
            i.destinationOperation = "{row['DESTINO DA OPERAÇÃO']}",
            i.isFinalConsumer = "{row['CONSUMIDOR FINAL']}",
            i.buyerPresence = "{row['PRESENÇA DO COMPRADOR']}",
            i.latestEvent = "{row['EVENTO MAIS RECENTE']}"

            MERGE (p:party {{ identifier: "{row['CPF/CNPJ Emitente']}" }})
            ON CREATE SET p.identifier = "{row['CPF/CNPJ Emitente']}",
            p.name = "{row['RAZÃO SOCIAL EMITENTE']}",
            p.stateRegistration = "{str(row['INSCRIÇÃO ESTADUAL EMITENTE'])}",
            p.type = "EMITTER"

            MERGE (i)-[:ISSUED_BY]->(p)

            MERGE (s:state {{ uf: "{row['UF EMITENTE']}" }})
            ON CREATE SET s.uf = "{row['UF EMITENTE']}"

            MERGE (p)-[:LOCATED_IN]->(s)

            MERGE (p2:party {{ identifier: "{str(row['CNPJ DESTINATÁRIO'])}" }})
            ON CREATE SET p2.identifier = "{str(row['CNPJ DESTINATÁRIO'])}",
            p2.name = "{row['NOME DESTINATÁRIO']}",
            p2.recipientTaxIndicator = "{row['INDICADOR IE DESTINATÁRIO']}",
            p2.type = "RECIPIENT"
            ON MATCH SET p2.recipientTaxIndicator = "{row['INDICADOR IE DESTINATÁRIO']}",
            p2.type = "BOTH"

            MERGE (s2:state {{ uf: "{row['UF DESTINATÁRIO']}" }})
            ON CREATE SET s2.uf = "{row['UF DESTINATÁRIO']}"

            MERGE (i)-[:ISSUED_TO]->(p2)

            MERGE (p2)-[:LOCATED_IN]->(s2)
            '''

        summary = driver.execute_query(cypher, database_="neo4j").summary
        nodes_count = nodes_count + summary.counters.nodes_created
        time_count = time_count + summary.result_available_after
        relationships_count = relationships_count + summary.counters.relationships_created
        properties_count = properties_count + summary.counters.properties_set
        labels_count = labels_count + summary.counters.labels_added

        if (len(df_heads)-index)%(len(df_heads)/(len(df_heads)/20))==0:
            bar = int((len(df_heads)-index)/len(df_heads)*20)
            msg = bar * "●" + (20-bar) * "○"
            writer({"custom_key": {"node": "data_ingestion", "status": "Ingesting heads:" + msg}})

    # Loops through Item file and create Invoice Items and ProductService and its connections
    for index, row in df_items.iterrows():
    
        cypher = f'''
            MERGE (ii:invoiceItem {{ uniqueItemId: "{str(row['NÚMERO']) + "-" + str(row['NÚMERO PRODUTO'])}" }})
            ON CREATE SET ii.uniqueItemId = "{str(row['NÚMERO']) + "-" + str(row['NÚMERO PRODUTO'])}",
            ii.itemNumber = {str(row['NÚMERO PRODUTO'])},
            ii.quantity = {str(row['QUANTIDADE'])},
            ii.unit = "{row['UNIDADE']}",
            ii.unitPrice = {str(row['VALOR UNITÁRIO'])},
            ii.totalItemPrice = {str(row['VALOR TOTAL'])}

            MERGE (i:invoice {{number: {str(row['NÚMERO'])} }})

            MERGE (i)-[:HAS_ITEM]->(ii)

            MERGE (ps:productService {{ productServiceId: "{str(row['NÚMERO']) + "-" + str(row['NÚMERO PRODUTO'])}" }})
            ON CREATE SET ps.productServiceId = "{str(row['NÚMERO']) + "-" + str(row['NÚMERO PRODUTO'])}",
            ps.ncmshId = "{str(row['CÓDIGO NCM/SH'])}",
            ps.ncmshDescription = "{str(row['NCM/SH (TIPO DE PRODUTO)'])}",
            ps.description = "{str(row['DESCRIÇÃO DO PRODUTO/SERVIÇO'])}",
            ps.cfop = {str(row['CFOP'])}

            MERGE (ii)-[:REFERS_TO_PRODUCT]->(ps)
            '''    
        summary = driver.execute_query(cypher, database_="neo4j").summary
        nodes_count = nodes_count + summary.counters.nodes_created
        time_count = time_count + summary.result_available_after
        relationships_count = relationships_count + summary.counters.relationships_created
        properties_count = properties_count + summary.counters.properties_set
        labels_count = labels_count + summary.counters.labels_added


        if (len(df_items)-index)%(len(df_items)/(len(df_items)/20))==0:
            bar = int((len(df_items)-index)/len(df_items)*20)
            msg = bar * "●" + (20-bar) * "○"
            writer({"custom_key": {"node": "data_ingestion", "status": "Ingesting ITEMS:" + msg}})

    # Create normal indexes

    writer({"custom_key": {"node": "data_ingestion", "status": "Creating Indexes"}})

    indexes_count = 0
    cypher = "CREATE INDEX issueDateTime_idx IF NOT EXISTS FOR (n:invoice) ON (n.issueDateTime)"
    summary = driver.execute_query(cypher, database_="neo4j").summary
    indexes_count = indexes_count + summary.counters.indexes_added
    time_count = time_count + summary.result_available_after
    cypher = "CREATE FULLTEXT INDEX partyName_ftidx IF NOT EXISTS FOR (p:party) ON EACH [p.name]"
    summary = driver.execute_query(cypher, database_="neo4j").summary
    indexes_count = indexes_count + summary.counters.indexes_added
    time_count = time_count + summary.result_available_after
    cypher = "CREATE FULLTEXT INDEX productServiceDescription_ftidx IF NOT EXISTS FOR (ps:productService) ON EACH [ps.description, ps.ncmshDescription]"
    summary = driver.execute_query(cypher, database_="neo4j").summary
    indexes_count = indexes_count + summary.counters.indexes_added
    time_count = time_count + summary.result_available_after
    
    #Create vector embeddings with "text-embedding-ada-002" from OpenAI

    batch_size = 100
    batch_n = 1
    pd_batch = []
    with driver.session(database=DB_NAME) as session:
        # Fetch `ProductService` nodes
        result = session.run('MATCH (ps:productService) WHERE ps.embedding IS NULL RETURN ps.description AS description, ps.ncmshDescription AS NCMSHDescription')
        for record in result:
            description = record.get('description')
            NCMSHDescription = record.get('NCMSHDescription')

            if description is not None and NCMSHDescription is not None:
                pd_batch.append({
                    'description': description,
                    'NCMSHDescription': NCMSHDescription,
                    'to_encode': f'Description: {description} - NCM_SH Description: {NCMSHDescription}'
                })

            # Import a batch; flush buffer
            if len(pd_batch) == batch_size:
                import_batch(driver, pd_batch, batch_n)
                pd_batch = []
                n= (len(df_items)//batch_size)
                msg = (n-batch_n) * "●" + (batch_n) * "○"
                writer({"custom_key": {"node": "data_ingestion", "status": "Creating vector indexes:" + msg}})
                batch_n += 1

        # Flush last batch
        if len(pd_batch) > 0:
            import_batch(driver, pd_batch, batch_n)

    # Create vector index
    summary = driver.execute_query('''
        CREATE VECTOR INDEX productServiceDescriptions_vectIdx IF NOT EXISTS
        FOR (ps:productService)
        ON ps.embedding
        OPTIONS { indexConfig: {
        `vector.dimensions`: 3072,
        `vector.similarity_function`: 'cosine'
        }}
        ''', database_="neo4j").summary
    indexes_count = indexes_count + summary.counters.indexes_added
    time_count = time_count + summary.result_available_after
    
    # Fix states
    cypher = ""
    for key in Estados.keys():
        cypher += f'MERGE ({key.lower()}:state {{ uf: "{key}" }}) SET {key.lower()}.stateName = "{Estados[key]}"\n\n'
    driver.execute_query(cypher, database_="neo4j")    

    return {
        "files_ingested": [
            {"file": HEADER_FILE},
            {"file": ITEM_FILE}
            ],
        "labels": labels_count,
        "nodes": nodes_count,
        "node_properties": properties_count,
        "edges": relationships_count,
        "indexes": indexes_count,
        "ingestion_time": time_count
        }