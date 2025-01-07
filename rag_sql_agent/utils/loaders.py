from langchain_community.vectorstores import FAISS


def load_faiss_index(
    embedding_model, index_path="faiss_index", allow_dangerous_deserialization=False
):
    try:
        return FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )
    except FileNotFoundError:
        return FAISS([], embedding_model)
