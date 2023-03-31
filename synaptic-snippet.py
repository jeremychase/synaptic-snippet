import requests
import feedparser
import sys

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain

from langchain.docstore.document import Document


llm = OpenAI(temperature=0.8)

text_splitter = CharacterTextSplitter()

def fetch_rss_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        feed_data = feedparser.parse(response.text)
        
        # Check if the feed data is a valid RSS feed
        if feed_data.version != '':
            return feed_data.entries
        else:
            raise ValueError('The data returned is not a valid RSS feed.')
    else:
        raise ConnectionError(f'Error fetching data from URL: {response.status_code}')

def generate_summary(content):
    docs = [Document(page_content=content[:4500])] # 4500 is the max length of a prompt due to openai's limit
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    return chain.run(docs)

# Ensure a URL is passed in
if len(sys.argv) != 2:
    print("Usage:")
    print("")
    print("  python3 synsni.py <rss-url>")
    sys.exit(1)

# Fetch the RSS data
url = sys.argv[1]
rss_data = fetch_rss_data(url)

# Only do the first two results for now
for entry in rss_data[:5]:
    print("URL: " + entry.link)
    print("Summary:")
    print(generate_summary(entry.content[0].value))
    print("")