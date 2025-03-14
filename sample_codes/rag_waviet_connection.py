import os
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate.classes.config import Configure
import requests, json, os

# Best practice: store your credentials in environment variables
weaviate_url = "XXXX"
weaviate_api_key = "XXXX"


# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

# questions = client.collections.create(
#     name="Question",
#     vectorizer_config=Configure.Vectorizer.text2vec_weaviate(), # Configure the Weaviate Embeddings integration
#     generative_config=Configure.Generative.cohere()             # Configure the Cohere generative AI integration
# )

# resp = requests.get(
#     "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
# )
# data = json.loads(resp.text)

# questions = client.collections.get(name="Question")

# with questions.batch.dynamic() as batch:
#     for d in data:
#         batch.add_object({
#             "answer": d["Answer"],
#             "question": d["Question"],
#             "category": d["Category"],
#         })
#         if batch.number_errors > 10:
#             print("Batch import stopped due to excessive errors.")
#             break

# failed_objects = questions.batch.failed_objects
# if failed_objects:
#     print(f"Number of failed imports: {len(failed_objects)}")
#     print(f"First failed object: {failed_objects[0]}")

# response = questions.query.near_text(
#     query="biology",
#     limit=2
# )

# for obj in response.objects:
#     print(json.dumps(obj.properties, indent=2))


questions = client.collections.get("Question")

response = questions.query.near_text(
    query="glycogen",
    limit=2,
    return_metadata=wvc.query.MetadataQuery(distance=True, certainty=True, score=True, explain_score=True)
)

# for obj in response.objects:
#     print(json.dumps(obj.properties, indent=2))



print(json.dumps(response, indent=2))

client.close()  # Free up resources
