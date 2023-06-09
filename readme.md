# synaptic-snippet

synaptic-snippet is a powerful and innovative tool that reads RSS feeds and generates concise, engaging tweets based on the content. By leveraging GPT technology, synaptic-snippet creates coherent and relevant tweets, allowing you to share the latest news and updates with your audience in an easily digestible format.

## Features

- Seamless integration with RSS feeds
- GPT-powered tweet generation for engaging content
- Automatic content summarization
- Customizable tweet formatting
- Easy integration with popular social media platforms

## Getting Started

Development is performed using [VSCode Development containers](https://code.visualstudio.com/docs/devcontainers/containers).

1. Install plugins
1. Populate `.devcontainer/devcontainer.env`

    ```
    echo OPENAI_API_KEY=YOUR_KEY > .devcontainer/devcontainer.env

    ```

1. Open in container
1. Edit code
1. Run flask:

    ```
    cd main
    flask --app main:app run --debug
    ```

### requirements.txt

Run:

```
pipreqs .
```