# Atividade-Obrigatoria-2025-06-18
Entrega atividade obrigatória curso i2a2 de Agentes Autônomos

## A framework escolhida: LangChain/LangGraph/LangSmith
LangChain e LangGraph são frameworks projetados para o desenvolvimento de aplicações impulsionadas por LLMs. Embora trabalhem juntos e sejam parte do mesmo ecossistema, eles têm focos ligeiramente diferentes:
**LangChain** é um framework de código aberto que simplifica o desenvolvimento de aplicações usando LLMs. Ele fornece uma interface padronizada e modular para interagir com modelos de linguagem e outros componentes, como fontes de dados externas e memória. É ideal para construir aplicações LLM com fluxos de trabalho lineares e bem definidos, como chatbots simples, sistemas de perguntas e respostas e geradores de conteúdo.
O **LangGraph** é uma extensão do LangChain, construída para lidar com cenários mais complexos e dinâmicos, especialmente no desenvolvimento de agentes de IA com estado (state-machines). Ele permite que você defina interações de agente como um grafo com estado persistente. Pode ser a escolha preferida quando se precisa construir agentes de IA que exigem interações cíclicas, processos de tomada de decisão complexos, colaboração entre múltiplos agentes e gerenciamento robusto de estado.
O **LangSmith** é uma plataforma muito interessante para o desenvolvimento e a otimização de aplicações que utilizam Modelos de Linguagem de Grande Escala (LLMs), especialmente aquelas construídas com LangChain e LangGraph. Ele atua como uma ferramenta de observabilidade e avaliação, preenchendo a lacuna entre a prototipagem e a produção de aplicações de IA robustas.

## Como a solução foi estruturada:
Testamos dois *approaches* diferentes. Começamos com um agente relativamente simples e enxuto, contendo apenas um orquestrador e um executor de ferramentas, usando ferramentas simples (@tools) um agente  do próprio framework (create_csv_agent). O agente recebe a pergunta do usuário, localiza o arquivo zip no diretório de execução, descompacta o arquivo em uma pasta, identifica os arquivos csv e os disponibiliza para um agente RAG csv que procura responder a pergunta da melhor maneira possível. A solução funciona a contento porém possui poucos recursos para aprimoramento, e melhorias. Por isso procurou-se explorar outra solução.
O segundo agente trabalha de forma iterativa com o usuário, como um chat e funciona com um fluxo mais elaborado o que lhe permite agir com maior flexibilidade e personalização. Os inputs do usuário passam sempre por um ‘node’ que identifica o estado de execução da tarefa e decide, às vezes de forma programática, às vezes por meio de uma LLM, qual a próxima ação a ser tomada.

### Agente 1. LangGraphAgent: solução básica
#### arquivo: ./scripts/LangGraphAgent.ipynb e .py
Como apresentado na figura a seguir, são apenas dois elementos. O orquestrador (main_assistant) recebe o input e decide como e quando usará as ferramentas. O executor de tarefas (tool_node) realiza a tarefa com a ferramenta que o orquestrador lhe pede e com os argumentos fornecidos pelo mesmo. 
Foram necessárias apenas três ferramentas:
1. Encontrar o nome do arquivo .zip na pasta do projeto (biblioteca os)
2. Descompactar o arquivo zip e encontrar o nome dos arquivos .csv (biblioteca zipfile)
3. Uma agente RAG do próprio LangChain (langchain_experimental.agents. .agent_toolkits.create_csv_agent)

![image](https://github.com/user-attachments/assets/0f5429f1-b844-4b37-8155-beeadecbf15a)

A seguir vemos um exemplo das etapas seguidas pelo agente para responder a uma pergunta:

```
Chamada:
agent1.nice_call("Quantas notas fiscais podem ser encontradas na base de dados?")


Saída:
--- AGENT EXECUTION LOG ---
--- Message 1 ---
  🤖 **System Message:**
    Content: You are a helpful assistant. Your job is to answer questions related to the contents of one or more CSV files
    that you have to extract from a .zip file. So before answering any question you need to find the zip file
    available to you, extarct it and only then you can try to answer the user inquiry.
--- Message 2 ---
  👤 **Human Message:**
    Content: Quantas notas fiscais podem ser encontradas na base de dados?
--- Message 3 ---
  🧠 **AI Message (Thought/Action):**
    Tool Calls:
      - Name: list_available_zip_files
        Arguments: {}
        Tool Call ID: call_hqwVn5a8OdvHrz11aQ5CQs6q
    Finish Reason: tool_calls
--- Message 4 ---
  🛠️ **Tool Message (Tool Output):**
    Tool Call ID: call_hqwVn5a8OdvHrz11aQ5CQs6q
    Content: {'zip_files': ['CSV_FILES.zip'], 'intermediate_steps': ['list_available_zip_files']}
--- Message 5 ---
  🧠 **AI Message (Thought/Action):**
    Tool Calls:
      - Name: extract_zip_files
        Arguments: {'local_zip_filepaths': ['CSV_FILES.zip']}
        Tool Call ID: call_H1E3Vsi5yb9cRTJ7al6VDKYl
    Finish Reason: tool_calls
--- Message 6 ---
  🛠️ **Tool Message (Tool Output):**
    Tool Call ID: call_H1E3Vsi5yb9cRTJ7al6VDKYl
    Content: {'csv_files': ['./datasets\\202401_NFs_Cabecalho.csv', './datasets\\202401_NFs_Itens.csv'], 'intermediate_steps': ['extract_zip_files']}
--- Message 7 ---
  🧠 **AI Message (Thought/Action):**
    Tool Calls:
      - Name: answer_question
        Arguments: {'question': 'Quantas notas fiscais podem ser encontradas na base de dados?', 'csv_files': ['./datasets\\202401_NFs_Cabecalho.csv', './datasets\\202401_NFs_Itens.csv']}
        Tool Call ID: call_XfPmNGrtKnPyd2eF3KQWgHcp
    Finish Reason: tool_calls
--- Message 8 ---
  🛠️ **Tool Message (Tool Output):**
    Tool Call ID: call_XfPmNGrtKnPyd2eF3KQWgHcp
    Content: {'final_answer': 'There are 200 unique invoices in the base of data.', 'intermediate_steps': ['answer_question']}
--- Message 9 ---
  🧠 **AI Message (Thought/Action):**
    Content: A base de dados contém 200 notas fiscais únicas.
    Finish Reason: stop

--- Summary ---
👣 **Intermediate Steps (Node Execution Order):**
  main_agent -> list_available_zip_files -> main_agent -> extract_zip_files -> main_agent -> answer_question -> main_agent
📦 **Identified Zip Files:** CSV_FILES.zip
📄 **Extracted CSV Files:**
  - ./datasets\202401_NFs_Cabecalho.csv
  - ./datasets\202401_NFs_Itens.csv
✅ **Final Answer:** There are 200 unique invoices in the base of data.

--- END OF AGENT LOG ---
```

### Agente 2. GraphRAGAgent: RAG GraphDB
#### arquivo: ./scripts/GraphRAGAgent.py
Como pode ser visto na figura a seguir, o agente é bastante mais complexo porém prevê maior grau de flexibilidade, personalização e melhoria de performance. Composto de um componente central que coordena a execução da tarefa (flowmanager) conforme interações com o usuário (human_input) e com demais assistentes. 
A primeira etapa do fluxo total é identificar o nome do arquivo zip (verify_file) e busca-lo na pasta google drive do i2a2 para o download e a extração dos arquivos csv (dload_n_xtract). Fetia a extação o agente passa automaticamente para a etapa de ingestão dos dados em uma base Neo4j (data_ingestion) quando cria também índices de busca textuais, por datas e também um incide semântico (vector embeddings) para os campos de produtos e serviços das notas fiscais.
Executadas estas tarefas de preparação o agente entra no ciclo de inquisição aos dados, interagindo com o usuário que realiza suas perguntas. Identificada uma pergunta ao usuário (get_inquiry) o agente passa a pergunta por uma verificação de pertinência (guardrails) de onde pode ou recusar a resposta ou passar para a geração do código CYPHER que faz o query na base de dados.
O ciclo de geração (generate_cypher), validação (validate_cypher), correção (correct_cypher) e execução (execute_cypher) da consulta CYPHER ocorre automaticamente entre assistentes especialistas, até que uma consulta seja aprovada e a extração de dados seja passada ao assistente que gera a resposta final (generate_final_answer).  No caso de três tentativas (configurável) de pesquisas CYPHER errôneas, o agente oferece ao usuário a opção de um CYPHER manual.
Com a resposta final voltamos à interação humana para a realização de mais perguntas, para se buscar um novo arquivo no drive da i2a2 ou terminando-se a execução do agente.

![image](https://github.com/user-attachments/assets/47e8da3b-0a3e-4032-bdfb-8472fb750f79)

A seguir vemos um exemplo das etapas seguidas pelo agente para responder a uma pergunta:
```
Chamada:
state_output = await agent2.call("Olá, bom dia")

Saída:
{'node': 'verify_file', 'status': 'Verifying file to download'}
{'node': 'verify_file', 'status': 'Identifying file name'}
================================ Human Message =================================
Olá, bom dia
================================== Ai Message ==================================
Olá, bom dia! Sou seu assistente virtual e estou aqui para ajudar com os dados de faturas de serviços públicos. Posso fazer o download de um arquivo zip da pasta do Google Drive do i2a2 para você. Para prosseguirmos, por favor, informe o nome do arquivo que deseja baixar.

Chamada:
state_output = await agent2.call("Ah sim, por favor, fazer o download o arquivo NF_202401.zip")

Saída:
{'node': 'verify_file', 'status': 'Verifying file to download'}
{'node': 'verify_file', 'status': 'Verifying file existance in drive'}
================================ Human Message =================================
Ah sim, por favor, fazer o download o arquivo NF_202401.zip
================================== Ai Message ==================================
Obrigado. Tentei encontrar o arquivo NF_202401.zip, mas não consegui. Você pode, por favo verificar e confirmar nome correto?

Chamada:
state_output = await agent2.call("Desculpe, realmente me enganei. O arquivo correto é 202401_NFs.zip")

Saída:
{'node': 'verify_file', 'status': 'Verifying file to download'}
{'node': 'verify_file', 'status': 'Verifying file existance in drive'}
{'node': 'dload_n_xtrack', 'status': 'Downloading zip file'}
{'node': 'dload_n_xtrack', 'status': "Successfull download and extraction of files ['202401_NFs_Cabecalho.csv', '202401_NFs_Itens.csv'] ."}
{'node': 'data_ingestion', 'status': 'Ingesting heads:●●●●●●●●●●●●●●●●●●●●'}
...
{'node': 'data_ingestion', 'status': 'Ingesting heads:●●●●○○○○○○○○○○○○○○○○'}
{'node': 'data_ingestion', 'status': 'Ingesting ITEMS:●●●●●●●●●●●●●●●●●●●○'}
...
{'node': 'data_ingestion', 'status': 'Ingesting ITEMS:●○○○○○○○○○○○○○○○○○○○'}
{'node': 'data_ingestion', 'status': 'Creating Indexes'}
{'node': 'get_inquiry', 'status': 'identifying question'}
{'node': 'get_inquiry', 'status': 'Failed to identify question, waiting for user prompt'}
================================[1m Human Message [0m=================================
Desculpe, realmente me enganei. O arquivo correto é 202401_NFs.zip
==================================[1m Ai Message [0m==================================
Obrigado por me passar o nome do arquivo. Consegui confirmar a identificação do arquivo 202401_NFs.zip. Vou prosseguir com o download, extração e ingestão.
==================================[1m Ai Message [0m==================================
CSV files ingested
==================================[1m Ai Message [0m==================================
Olá! Com base na estrutura do banco de dados, posso ajudá-lo a consultar informações relacionadas a notas fiscais, empresas, produtos, estados e itens de nota fiscal. Você pode solicitar dados como detalhes de uma nota específica, informações sobre uma empresa, produtos mais vendidos, ou dados por estado. Por favor, diga qual informação você deseja ou formule sua pergunta!

Chamada:
state_output = await agent2.call("Quantas notas fiscais temos na base de dados?")

Saída:
{'node': 'get_inquiry', 'status': 'identifying question'}
{'node': 'guardrail', 'status': 'Testing guardrails'}
{'node': 'generate_cypher', 'status': 'generating cypher query'}
{'node': 'validate_cypher', 'status': 'checking for cypher statement errors'}
{'node': 'execute_cypher', 'status': 'Excetute query'}
{'node': 'generate_final_answer', 'status': 'Generating Final answer'}
================================ Human Message =================================
Quantas notas fiscais temos na base de dados?
================================ Human Message =================================
```

## Perguntas e respostas
#### Arquivo testes: ./scripts/Perguntas_e_Respostas.ipynb
Elaboramos 5 perguntas para as quais as respostas poderiam ser encontradas na base de dados com o objetivo de testar a capacidade de cada modelo encontrar as respostas corretas. Quadro abaixo podemos verificar os resultados:

```
PERGUNTA 01: Quantas notas fiscais estão na base de dados?
AGENTE 1 - There are 200 unique invoices in the database.
AGENTE 2 - Na base de dados, há um total de 100 notas fiscais.

RESPOSTA CORRETA: Temos 100 notas fiscais na base de dados
------------------------------
PERGUNTA 02: Qual o valor total das notas emitidas entre os dias 10 de Janeiro de 2024 e 20 de Janeiro de 2024
AGENTE 1 - 1,340,519.56
AGENTE 2 - O valor total das notas emitidas entre os dias 10 de Janeiro de 2024 e 20 de Janeiro de 2024 é de R$ 1.340.519,56.

RESPOSTA CORRETA: Neste período, foram emitidas notas que somam o valor de R$ 1,340,519.56
------------------------------
PERGUNTA 03: Qual a empresa que mais fez compras (recebeu notas) no estado do Paraná?
AGENTE 1 - According to the data provided, there are no recipients (companies) that received invoices in the state of Paraná.
AGENTE 2 - A empresa que mais fez compras (recebeu notas) no estado do Paraná é a UNIVERSIDADE FEDERAL DO PARANÁ, com um total faturado de R$ 7.486,50.

RESPOSTA CORRETA: A empresa, no estado do Paraná, que mais recebeu emissão de notas foi a UNIVERSIDADE FEDERAL DO PARANÀ
------------------------------
PERGUNTA 04: Alguma empresa comprou alimentos?
AGENTE 1 - Sim, algumas empresas compraram alimentos. Identificamos a "FUNDACAO UNIVERSIDADE FEDERAL DE MS" e "SUPERINTENDENCIA REGIONAL SUDESTE I" comprando "AGUA MINERAL NATURAL".
AGENTE 2 - Sim, várias empresas compraram alimentos. As informações mostram que a UNIVERSIDADE FEDERAL DO RIO GRANDE NORTE comprou alho branco, o COLEGIO MILITAR DE PORTO ALEGRE adquiriu tomate salada, e o MINISTÉRIO DO DESENVOLVIMENTO E ASSISTÊNCIA SOCIAL, FAMÍLIA comprou quiabo e batata doce. Além disso, o MINISTÉRIO DO DESENVOLVIMENTO SOCIAL, FAMÍLIA E COMBATE À FOME também adquiriu batata doce.

RESPOSTA CORRETA: Sim. Algumas empresas compraram produtos alimentícios, como pão francês, tomate salada, quiabo e batata doce.
------------------------------
PERGUNTA 05: Qual o valor total das notas emitidas sobre materiais escolares?
AGENTE 1 - The total value of invoices emitted for school materials is 531.82.
AGENTE 2 - O valor total das notas emitidas sobre materiais escolares, com base nos resultados fornecidos, é de R$ 118.000,00. Esse valor é a soma dos preços dos itens relacionados a materiais escolares, como livros e outros suprimentos.

RESPOSTA CORRETA: Parece que foram gastos cerca de R$ 294,673.35 com materiais escolares
------------------------------
```

## Conclusões:
 Obviamente que uma avaliação rigorosa poderia ser executada com mais questionamentos e, principalmente, diversas rodadas de respostas, visto que a natureza probabilística dos modelos faz com que suas respostas não sejam sempre as mesmas. Independentemente disso, vemos claramente que o segundo modelo, apesar de muito mais complexo e com muitas mais linhas de código que o primeiro (1.161 VS 265) é não apenas mais capaz de responder satisfatoriamente a perguntas complexas, mas também, no permitiria ajustes e aprimoramento detalhado de cada etapa do agente. 
Enquanto o modelo LangGraph padrão pode nos fornecer uma solução rápida e barata para a construção de agentes que utilizam ferramentas e consultam bases de dados, o modelo personalizado nos permite, potencialmente, mergulhar muito mais fundo em estruturas de dados complexas (grafos), personalização total das tarefa e da interação com o usuário (human-in-the-loop) e, também potencialmente, ajustes finos de prompt engineering, das lógicas e métodos para a criação, verificação e correção dos queries cypher, e muito mais.






