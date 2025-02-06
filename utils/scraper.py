from langchain_community.document_loaders import WebBaseLoader

def load_website_data(url):
    """Loads and processes text data from a website."""
    loader = WebBaseLoader(url)
    return loader.load()
