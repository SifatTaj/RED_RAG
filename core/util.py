import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_yaml(path):
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

def make_chunks(doc_path:str,
                chunk_size:int,
                chunk_overlap:int
                ) -> list[str]:

    chunks = []
    with open(doc_path) as f:
        doc = f.read()
        ts = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        chunks = ts.split_text(doc)
    return chunks