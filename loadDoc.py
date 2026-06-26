from langchain_docling.loader import DoclingLoader

FILE_PATH = "./docs/rules.pdf"

loader = DoclingLoader(file_path=FILE_PATH)

documents = loader.load()


