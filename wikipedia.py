# https://zenn.dev/articles/f13d1246279fa2
# WikipediaのIntegrationを利用したRAGを行います

# 必要なライブラリを取得
from langchain_community.llms import Cohere
from langchain.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# WikipediaRetrieverを設定
retriever = WikipediaRetriever()

# LLMが出力するデータの形をstring型に整形します。
output_parser = StrOutputParser()

# Prompt,llmを設定
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the context provided:

Context: {context}

Question: {question}""")

llm = Cohere(model="command", cohere_api_key="自分のAPI KEY")

chain = prompt | llm | output_parser

# RunnablePassthroughを設定
retrieval_chain = RunnablePassthrough.assign(
    context=(lambda x: x["question"]) | retriever
) | chain

retrieval_chain.invoke({"question": "what is ONEPIECE?"})