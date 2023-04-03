import requests
import feedparser
import sys
import os

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain

from langchain.docstore.document import Document

# Cache responses
import langchain
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# Flask
from flask import Flask, abort, jsonify, make_response, redirect, request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

# html updating
from bs4 import BeautifulSoup

swagger_template = dict(
info = {
    'title': LazyString(lambda: 'Synaptic Snippet'),
    'version': LazyString(lambda: '0.1'),
    'description': LazyString(lambda: 'An <a href="https://github.com/jeremychase/synaptic-snippet">open source</a> api for building tweets from an RSS feed using Langchain and OpenAI.'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/api.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

def app(environ, start_response):
    data = b"Hello, World!\n"
    start_response("200 OK", [
        ("Content-Type", "text/plain"),
        ("Content-Length", str(len(data)))
    ])
    return iter([data])

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


def parse_tweets(input_str):
    # Split the input string into lines
    lines = input_str.splitlines()

    # Reverse the lines so that we can iterate from the end
    reversed_lines = reversed(lines)

    # Initialize a list to store the non-empty lines
    non_empty_lines = []

    # Iterate through the reversed lines until we have found three non-empty lines
    for line in reversed_lines:
        # If the line is non-empty and we have found three non-empty lines already, break the loop
        if line.strip() and len(non_empty_lines) == 3:
            break
        # If the line is non-empty, add it to the list of non-empty lines
        elif line.strip():
            non_empty_lines.append(line.strip())

    # Reverse the list of non-empty lines so that they are in their original order
    return list(reversed(non_empty_lines))


def generate_tweets(rss_data, temperature, twitter_handle):
    input = [None] * len(rss_data)
    output = []
    twitter_style = ""

    if twitter_handle != None:
        twitter_style = f", in the style of twitter user {twitter_handle}"

    for i in range(0, len(rss_data)):
        input[i] = f"Generate three catchy tweets{twitter_style}, one on each line, with no label, linking to {rss_data[i].link} near the end, based on:\n\n {remove_html_tags(rss_data[i].content[0].value)}"
        input[i] = input[i][:4500] # TODO check on truncation

        output.append({
            "link": rss_data[i].link,
            "title": rss_data[i].title
        })

    llm = OpenAI(temperature=temperature)
    llm_result = llm.generate(input)

    for i in range(0, len(llm_result.generations)):
        output[i]["tweets"] = parse_tweets(llm_result.generations[i][0].text)

    return output


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=' ')
    return stripped_text


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # app.config.from_mapping(
    #     SECRET_KEY='dev',
    #     DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    # )

    # Setup Swagger
    swagger = Swagger(app, template=swagger_template,             
                    config=swagger_config)

    app.json_encoder = LazyJSONEncoder


    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/")
    def docs_redirect():
        return redirect(swagger_config['specs_route'], code=302)


    @swag_from("tweets.yml", methods=['GET'])
    @app.route("/tweets")
    def tweets():
        feed_url = request.args.get("feed_url")
        temperature = request.args.get("temperature")
        twitter_handle = request.args.get("twitter_handle")
        msg = None

        # vet temperature param
        if temperature == None:
            temperature = 0.7
        else:
            temperature = float(temperature)
            if temperature > 2.0 or temperature < 0.0:
                msg = 'invalid temperature query parameter, must be between 0.0 to 2.0'

        # vet feed_url param
        if feed_url == None:
            msg = 'missing feed_url query parameter'
        elif len(feed_url) <= 0:
            msg = 'invalid feed_url'

        if msg != None:
            response = make_response(
                jsonify(
                    {"message": msg, "status": "user-error"}
                ),
                400,
            )

            response.headers["Content-Type"] = "application/json"
            return response


        rss_data = fetch_rss_data(feed_url)

        tweets = generate_tweets(rss_data, temperature, twitter_handle)
        
        response = make_response(
            jsonify(
                {
                "tweets": tweets
                }
            ),
            200,
        )

        response.headers["Content-Type"] = "application/json"
        return response

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
