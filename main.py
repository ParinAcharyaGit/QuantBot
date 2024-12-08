import streamlit as st
from langchain_pinecone import PineconeVectorStore
import dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
import openai

def get_stock_info(symbol:str) -> dict:
    data = yf.Ticker(symbol)
    stock_info = data.info

    properties = {
        "Ticker": stock_info.get('symbol', 'Information not available'),
        'Name': stock_info.get('longName', 'Information not available'),
        'Business Summary': stock_info.get('longBusinessSummary'),
        'City': stock_info.get('city', 'Information not available'),
        'State': stock_info.get('state', 'Information not available'),
        'Country': stock_info.get('country', 'Information not available'),
        'Industry': stock_info.get('industry', 'Information not available'),
        'Sector': stock_info.get('sector', 'Information not available')
    }

    return properties

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

def get_cosine_similarity(s1, s2):
     # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(s1))
    embedding2 = np.array(get_huggingface_embeddings(s2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    similarity_score = similarity[0][0]
    print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
    return similarity_score

s1 = "I like walking to the park"
s2 = "I like runnning to the office"
similarity = get_cosine_similarity(s1, s2)
print(similarity)


def get_company_tickers():
    url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"

    # Making a GET request to the URL
    response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parse the JSON content directly
        company_tickers = json.loads(response.content.decode('utf-8'))

        # Optionally save the content to a local file for future use
        with open("company_tickers.json", "w", encoding="utf-8") as file:
            json.dump(company_tickers, file, indent=4)

        print("File downloaded successfully and saved as 'company_tickers.json'")
        return company_tickers
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None
    
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = "stocks"
namespace = "stock-descriptions"
hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

# Initialize tracking lists
successful_tickers = []
unsuccessful_tickers = []

# Load existing successful/unsuccessful tickers
try:
    with open('successful_tickers.txt', 'r') as f:
        successful_tickers = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(successful_tickers)} successful tickers")
except FileNotFoundError:
    print("No existing successful tickers file found")

try:
    with open('unsuccessful_tickers.txt', 'r') as f:
        unsuccessful_tickers = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(unsuccessful_tickers)} unsuccessful tickers")
except FileNotFoundError:
    print("No existing unsuccessful tickers file found")

def process_stock(stock_ticker: str) -> str:
    # Skip if already processed
    if stock_ticker in successful_tickers:
        return f"Already processed {stock_ticker}"

    try:
        # Get and store stock data
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']

        # Store stock description in Pinecone
        vectorstore_from_texts = PineconeVectorStore.from_documents(
            documents=[Document(page_content=stock_description, metadata=stock_data)],
            embedding=hf_embeddings,
            index_name=index_name,
            namespace=namespace
        )

        # Track success
        with open('successful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        successful_tickers.append(stock_ticker)

        return f"Processed {stock_ticker} successfully"

    except Exception as e:
        # Track failure
        with open('unsuccessful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        unsuccessful_tickers.append(stock_ticker)

        return f"ERROR processing {stock_ticker}: {e}"

def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
    # Only process up to 10 stocks
    tickers = tickers[:10]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)

                # Stop on error
                if result.startswith("ERROR"):
                    print(f"Stopping program due to error in {ticker}")
                    executor.shutdown(wait=False)
                    raise SystemExit(1)

            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
                print("Stopping program due to exception")
                executor.shutdown(wait=False)
                raise SystemExit(1)

st.header("Quantbot")
st.write("Automated financial analysis to gain a competitive edge in investments")

stock_finder = st.button("Query stocks")
company_tickers = get_company_tickers()

tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]
# Process them
parallel_process_stocks(tickers_to_process, max_workers=10) # Change the logic to only query stocks relevant to input and retrive tickers ONCE from Pinecone
# Prepare your tickers
pc = Pinecone(api_key = pinecone_api_key)
pinecone_index = pc.Index(index_name)
query = st.text_input("Enter your query...")
if query and st.button("Enter"):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    print("completed augmented query")
    openai.api_base = "https://api.groq.com/openai/v1"
    openai.api_key = os.environ.get("GROQ_API_KEY")
    print("initialised openai")

    system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.
    Provide a maximum of 5 related tickers.
    Limit your response in this structured format, like:
    Tickers: ['AAPL', 'MSFT', 'GOOGL'],
    Explanation: <explanation of why you chose each of these tickers>
    """

    llm_response = openai.ChatCompletion.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    st.write(response)

    st.header("Market Firehose - Related articles")
    # Function - depending on the user's query, return relevant articles and link them.
    from bs4 import BeautifulSoup
    import http.client, urllib.parse

    conn = http.client.HTTPSConnection('api.thenewsapi.com')

    params = urllib.parse.urlencode({
    'api_token': 'TFp9I8aFM42JBthMLpKdWNbRluZQ25rfQH8ttenQ',
    'sort': 'published_on',
    'limit': 3
    })

    conn.request('GET', '/v1/news/all?{}'.format(params))

    res = conn.getresponse()
    data = res.read()

    refined_data = data.decode('utf-8')
    print(refined_data)

    system_prompt = f"""You are an expert financial news analyst. 
    You analyze real-time text articles given to you as {refined_data}. 
    Using that as context, filter only relevant articles related to the company stocks in {response}
    You must then evaluate how these filtered articles would affect the company stocks.
    In your response, mention the following in structured format for each article:
    - <title of the article>,
    - <a URL to the article>,
    - <short explanation of the company ticker is affected by the news in the article>,
    - <sentiment of the article, rated out of 10>,
    - <market cap of the company>    
    
    """

    article_response = openai.ChatCompletion.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": response}
        ]
    )

    article_response = article_response.choices[0].message.content
    st.write(article_response)











