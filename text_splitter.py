from langchain.text_splitter import  CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# random_text = (
#     "## Section 1\n\n"
#     "Paragraph one. It has multiple sentences. Some are short. Some are longer, with commas and clauses.\n\n"
#     "## Section 2\n\n"
#     "Bullet points:\n"
#     "- First point is brief.\n"
#     "- Second point is a bit longer and contains more detail, so it helps the splitter find natural breaks.\n\n"
#     "## Section 3\n\n"
#     "A very long paragraph without newlines but with periods. " * 20
# )

# character_splitter = CharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=0,
#     separator=""
# )

# character_splitter.create_documents([random_text])
# chunks = character_splitter.split_text(random_text)

# print(len(chunks))
# print(chunks[3])


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# recursive_character_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=0,
#     length_function=len
# )

# recursive_character_splitter.create_documents([random_text])
# chunks = recursive_character_splitter.split_text(random_text)
# print(len(chunks))
# print(chunks[3])



from PyPDF2 import PdfReader
reader = PdfReader("Study Guide_MSc_CS2020.pdf")
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + "\n"
print(len(raw_text))
pdf_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator=""
)
pdf_splitter.create_documents([raw_text])
chunks = pdf_splitter.split_text(raw_text)
print(len(chunks))
print(len(chunks[3]))

pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
pdf_splitter.create_documents([raw_text])
chunks = pdf_splitter.split_text(raw_text)
print(len(chunks))
print(len(chunks[3]))
