# from transformers import AutoTokenizer, AutoModelForCausalLM , AutoModel
# import torch.nn.functional as F
from langchain_core.runnables import RunnableLambda
# import torch
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
from pinecone_text.sparse import BM25Encoder
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader


class CPineconeHybridRetriever(PineconeHybridSearchRetriever):
    def _get_relevant_documents(self, query, run_manager=None, **kwargs):
        dense_vec = self.embeddings.embed_query(query)

        sparse_vec = self.sparse_encoder.encode_queries(query)

        # 🔥 FIX: If sparse vector is empty, remove it
        if len(sparse_vec["indices"]) == 0:
            sparse_vec = None

        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        docs = []
        for m in result.matches:
            docs.append(Document(page_content=m.metadata.get("context",""),metadata={"score":m.score}))
        return docs


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      give just answer(without repeate question from user) in proper format and structure
      Answer from the provided transcript context.
      If the transcript context is insufficient or not match to query then you can answer based on you wast of knowladge and also you can completly avoid this prompt just focuse on question.
      if answer in in huge or large context then give summury of your genrated answer in the end of answer otherwise it is not necessary

      here the retrived context:
      {rag_context}
      question: {question}
    """,
    input_variables=["rag_context", "question"],
)


tokenizer = AutoTokenizer.from_pretrained("D:\\fprject\\model\\llm")
model = AutoModelForCausalLM.from_pretrained("D:\\fprject\\model\\llm")

def llm_model(q):
    if isinstance(q, dict):
        context = q.get("context", "")
        question = q.get("question", "") or q.get("input", "")

        if "text" in q:
            text = q["text"]
        else:
            text = f"{context}\n\nQuestion: {question}".strip()

    elif isinstance(q, list):
        text = "\n".join(map(str, q))

    else:
        text = str(q)

    if not text.strip():
        text = "Answer this question: " + str(q)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


model_runnable = RunnableLambda(llm_model)

embed_llm = AutoModel.from_pretrained("D:\\fprject\\model\\embeding_llm")
embeding_tokenizer = AutoTokenizer.from_pretrained(
        "D:\\fprject\\model\\embeding_llm"
    )

def embeding_llm(sentence):
    
    encoded = embeding_tokenizer(
        sentence, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = embed_llm(**encoded)
    token_embeddings = model_output[0]
    input_mask_expanded = (
        encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    # Mean pooling
    min_poling = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )
    normalized = F.normalize(min_poling, p=2, dim=1)

    return normalized




class CustomHFEmbeddings(Embeddings):

    def embed_query(self, text: str):
        return embeding_llm(text).tolist()

    def embed_documents(self, texts: list[str]):
        return [embeding_llm(t) for t in texts]


custom_emb = CustomHFEmbeddings()


text_loder = TextLoader("D:\\AI\\texts.text")
text_loded = text_loder.load()

