from pymilvus import connections, utility

# Connessione a Milvus
connections.connect(
    alias="default",
    host="milvus.apps.eni.lajoie.de",  # o il tuo host
    port="80"                       # o la porta che usi
)

# Droppa tutta la collection
COLLECTION_NAME = "rag_chunks"
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' eliminata.")
else:
    print(f"Collection '{COLLECTION_NAME}' non trovata.")
