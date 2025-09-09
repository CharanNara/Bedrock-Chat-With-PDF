import boto3, json
from PyPDF2 import PdfReader

# boto3 - AWS SDK for Python
# pip install boto3

# Create Bedrock Runtime client for us-east-2 region
# Make sure your AWS credentials are configured in your environment
# e.g., via AWS CLI or environment variables
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# For more details, see: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
# Also, ensure that your IAM user/role has the necessary permissions to access Bedrock
# e.g., "bedrock:InvokeModel"   


# messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"text": "Tell me a joke about computers."}
#             ]
#         }
#     ]

# inference_config = {
#     "max_tokens": 2000,
#     "temperature": 1,
#     "top_p": 0.999,
#     "stopSequences": ["\n"]
# }

# additional_fields = {
#     "top_k": 250
# }

# performance_config = {
#     "latency": "standard"
# }

# response = client.converse(
#     modelId="anthropic.claude-3-5-sonnet-20240620",
#     messages=messages,
#     inferenceConfig=json.dumps(inference_config),
#     additionalModelRequestFields=json.dumps(additional_fields),
#     performanceConfig=json.dumps(performance_config)
# )


# Example with Claude 3.5 Sonnet to summarize text
# converse API reference: https://docs.aws.amazon.com/bedrock/latest/userguide/API_runtime_Converse.html
# converse API is used for chat-based models like Claude, Gemini, etc. 

#////////////////////////////////////////////////////////////////////////////////////
# response = client.converse(
#     modelId="arn:aws:bedrock:us-east-2:285982080176:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
#     messages=[
#         {"role": "user", 
#          "content": [
#             {
#                 "text": "Summarize the text: The Northern Lights \
# There are times when the night sky glows with bands of color. The bands may \
# begin as cloud shapes and then spread into a great arc across the entire sky. They \
# may fall in folds like a curtain drawn across the heavens. The lights usually grow \
# brighter, then suddenly dim. During this time the sky glows with pale yellow, pink, \
# green, violet, blue, and red. These lights are called the Aurora Borealis. Some \
# people call them the Northern Lights. Scientists have been watching them for \
# hundreds of years. They are not quite sure what causes them. In ancient times \
# Long Beach City College WRSC \
# people were afraid of the Lights. They imagined that they saw fiery dragons in the\
# sky. Some even concluded that the heavens were on fire."
#             }
#                     ]
#         }
#     ],
#     inferenceConfig={                 # <-- dict, not string
#         "maxTokens": 200,
#         "temperature": 1.0,
#         "topP": 0.999,
#     },
#     additionalModelRequestFields={    # <-- dict, not string
#         "top_k": 250
#     },
#     performanceConfig={               # <-- dict, not string
#         "latency": "standard"
#     }
# )

# print(response["output"]["message"]["content"][0]["text"])
# ////////////////////////////////////////////////////////////////////////////////////
# get embeddings from Bedrock
# embed API reference: https://docs.aws.amazon.com/bedrock/latest/userguide/API_runtime_Embed.html


import boto3, json
from PyPDF2 import PdfReader


client = boto3.client("bedrock-runtime", region_name="us-east-2")

# AWS Titan Embeddings model
from langchain_community.embeddings import BedrockEmbeddings

# 1. Bedrock runtime client (region where Titan v2 is enabled for you)
client = boto3.client("bedrock-runtime", region_name="us-east-2")

# 2. Bedrock Embeddings wrapper
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=client
)


# 3. Load and split text
reader = PdfReader("Study Guide_MSc_CS2020.pdf")
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + "\n"
    

from langchain.text_splitter import RecursiveCharacterTextSplitter
# split text into smaller chunks (e.g., 1000 tokens with 200 token overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

chunks = text_splitter.split_text(raw_text)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk length: {len(chunks[0])}")

# 4. Get embeddings
# embedding_vector = bedrock_embeddings.embed_documents([text])

# print("Vector length:", len(embedding_vector[0]))
# print("First 10 dims:", embedding_vector[0][:10])

from langchain_community.vectorstores import FAISS # also need to install faiss-cpu package
# create a vector store from the text and its embedding
vector_search_saved = FAISS.from_texts(texts=chunks, embedding=bedrock_embeddings)
vector_search_saved.save_local("faiss_StudyGuide")

# load the vector store
vector_search = FAISS.load_local("faiss_StudyGuide", embeddings=bedrock_embeddings, allow_dangerous_deserialization=True)

query = "What is the conclusion in this PDF?"
docs = vector_search.similarity_search(query, k=10)
# print(docs[0].page_content)
print(docs)


context = "\n".join([doc.page_content for doc in docs])


PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end to provide a concise and accurate answer.
Atleast summarize with 100 words with detailed explanation. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
Answer in markdown format with detailed explanation.
"""

# using claude-3-5-sonnet to generate answer from context
response = client.converse(
    modelId="arn:aws:bedrock:us-east-2:285982080176:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[
        {"role": "user", 
         "content": [
            {
                "text": PROMPT_TEMPLATE.format(context=context, question=query)
            }
                    ]
        }
    ],
    inferenceConfig={                 # <-- dict, not string
        "maxTokens": 1000,
        "temperature": 1.0,
        "topP": 0.999,
    },
    additionalModelRequestFields={    # <-- dict, not string
        "top_k": 250
    },
    performanceConfig={               # <-- dict, not string
        "latency": "standard"
    }
)

print(response["output"]["message"]["content"][0]["text"])
# End of Bedrock embeddings example

