import weaviate
from weaviate.embedded import EmbeddedOptions
from google.cloud import storage
import pandas as pd
import json
import os
import google.generativeai as genai
import vertexai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part, SafetySetting


# Start an embedded Weaviate instance
weaviate_client = weaviate.Client(
embedded_options=EmbeddedOptions()
)

# Create a schema
weaviate_client.schema.create_class({
    "class": "Report",
    "properties": [
        {
            "name": "report_date",
            "dataType": ["string"]
        },
        {
            "name": "report_type",
            "dataType": ["string"]
        },
        #{
        #    "name": "id",          # "id" is not allowed by weaviate
        #    "dataType": ["string"]
        #},
        {
            "name": "document_id",
            "dataType": ["string"]
        },
        {
            "name": "chunk",
            "dataType": ["text"]
        },
        {
            "name": "chunk_number",
            "dataType": ["int"]
        },
        {
            "name": "company_name",
            "dataType": ["string"]
        },
        #{
        #    "name": "embedding",      # Should be left out in the schema
        #    "dataType": ["number[]"]  # Assuming embeddings are stored as arrays
        #}
    ]
})

# get the schema to make sure it worked
weaviate_client.schema.get()

## Load json files from google cloud into dataframe

# Authenticate using your service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# Initialize a client
storage_client = storage.Client()

# Define your bucket and prefix
bucket_name = 'financial-reports-embeddings'
prefix = 'financial_reports/'
bucket = storage_client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)

# Download and parse JSON files
data = []
#count = 0
for blob in blobs:
    if blob.name.endswith('.json'):
        content = blob.download_as_string()
        json_data = json.loads(content)
        data.append(json_data)
        #count += 1
        #if count == 100:
        #    break

# Convert to DataFrame
df = pd.json_normalize(data)

# Rename columns
df.columns = [
    'embedding', 'chunk', 'document_id', 'company_name',
    'report_type',  'report_date', 'chunk_number',
]

# Enforce data types
df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x)))
df['chunk'] = df['chunk'].astype(str)
df['document_id'] = df['document_id'].astype(str)
df['company_name'] = df['company_name'].astype(str)
df['report_type'] = df['report_type'].astype(str)
df['report_date'] = df['report_date'].astype(str)
df['chunk_number'] = df['chunk_number'].astype(int)

# Display the DataFrame
print(df.head())
print(df['embedding'][0])

my_variable = df['embedding'][0]
if isinstance(my_variable, list):
    print("This variable is a list.")
else:
    print("This variable is not a list.")

### Step 1 - configure Weaviate Batch, which optimizes CRUD operations in bulk
# - starting batch size of 100
# - dynamically increase/decrease based on performance
# - add timeout retries if something goes wrong

weaviate_client.batch.configure(
    batch_size=100,
    dynamic=True,
    timeout_retries=3,
)


### Step 2 - import data

print("Uploading data with vectors to schema..")

counter=0

with weaviate_client.batch as batch:
    for k,v in df.iterrows():

        # print update message every 100 objects
        if (counter %100 == 0):
            print(f"Import {counter} / {len(df)} ")

        properties = {
            "company_name": v["company_name"],
            "report_type": v["report_type"],
            "report_date": v["report_date"],
            "document_id": v["document_id"],
            "chunk_number": v["chunk_number"],
            "chunk": v["chunk"],
        }

        vector = v["embedding"]

        batch.add_data_object(properties, "Report", None, vector)
        counter = counter+1

print(f"Importing ({len(df)}) Embedding files complete")


def query_weaviate(query, collection_name, top_k=20):

    # Generate embedding for the query
    embedded_query = genai.embed_content(
        content=query,
        model="models/text-embedding-004"
    )['embedding']

    near_vector = {"vector": embedded_query}

    # Queries input schema with vectorised user query
    query_result = (
        weaviate_client.query
        .get(collection_name, ["chunk", "chunk_number", "_additional {certainty distance}"])
        .with_near_vector(near_vector)
        .with_limit(top_k)
        .do()
    )

    return query_result


def retrieve_relevant_documents(query, collection_name):
    query_result = query_weaviate(query, "Report")
    counter = 0
    for chunk in query_result["data"]["Get"]["Report"]:
        counter += 1
        print(f"{counter}. { chunk['chunk']} (Certainty: {round(chunk['_additional']['certainty'],3) }) (Distance: {round(chunk['_additional']['distance'],3) })")

    relevant_documents = [item['chunk'] for item in query_result["data"]["Get"]["Report"]]
    print(relevant_documents)

    return relevant_documents


def generate_response(documents, query):
  
    # Prepare context from documents
    context = "\n".join(documents)

    # Initialize Vertex AI
    vertexai.init(project="sincere-venture-434618-s5", location="asia-southeast1")

    # Create the generative model
    model = GenerativeModel("gemini-1.5-flash-001",
                            system_instruction=[""""""]
                            )
    chat = model.start_chat()


    # Define generation configuration
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    # Define safety settings
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]

    # Send message and print response
    response = chat.send_message(
        f"Context: {context}\n\nQuery: {query}",  # Combine context and query
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return response.text

# Example
query = ""
collection_name = "Report"
relevant_documents = retrieve_relevant_documents(query,collection_name)
response = generate_response(relevant_documents, query)
print("Response:", response)


