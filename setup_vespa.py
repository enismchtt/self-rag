from vespa.package import ApplicationPackage, Document as VespaDocument, Field, RankProfile, Schema, QueryProfile, QueryProfileType, QueryTypeField, HNSW
from vespa.deployment import VespaDocker

embedding_dim = 768 # for sentence-transformers/LaBSE
app_name = "selfrag" 

# Define document schema
doc = VespaDocument(
    fields=[
        Field(
            name="text",
            type="string",
            indexing=["summary", "index"],
            index="enable-bm25"
        ),
        Field(
            name="embedding",
            type=f"tensor<float>(x[{embedding_dim}])",
            indexing=["attribute", "index"],
            ann=HNSW(   # Hierarchical Navigable Small World algorithm 
                distance_metric="euclidean",
                max_links_per_node=16,  
                neighbors_to_explore_at_insert=500
            )
        )
    ]
)

# Define schema with semantic ranking
schema = Schema(
    name= app_name,
    document=doc,
    rank_profiles=[
        RankProfile(
            name="semantic-similarity",
            inherits="default",
            first_phase="closeness(embedding)"
        )
    ]
)

# Create application package with query profile
app_package = ApplicationPackage(
    name= app_name,
    schema=[schema],
    query_profile=QueryProfile(),
    query_profile_type=QueryProfileType(
        fields=[
            QueryTypeField(
                name=f"ranking.features.query(query_embedding)",
                type=f"tensor<float>(x[{embedding_dim}])"
            )
        ]
    )
)

# Deploy Vespa app
vespa_docker = VespaDocker()
vespa_docker.deploy(application_package=app_package)
print("🚀 Vespa application deployed successfully.")