import json
import pickle
from typing import List, Dict, Optional
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..models.config import VectorDBConfig


class VectorDBService:
    def __init__(self, config: VectorDBConfig, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.persist_directory = self.data_dir / ".chroma_db"

        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        self.vectorstore: Optional[Chroma] = None
        self.qa_pairs: List[Dict[str, str]] = []

    def index_exists(self) -> bool:
        return self.persist_directory.exists() and any(self.persist_directory.iterdir())

    def load_or_create_index(self, reindex: bool = False) -> None:
        if self.index_exists() and not reindex:
            self.vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            self._load_qa_pairs()
        else:
            self._create_index()

    def _create_index(self) -> None:
        documents = []
        self.qa_pairs = []

        print(f"Creating index for data directory: {self.data_dir}")
        print(f"Persist directory will be: {self.persist_directory}")

        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.data_dir.iterdir():
            if file_path.suffix == ".txt":
                content = file_path.read_text(encoding="utf-8")
                docs = self.text_splitter.create_documents(
                    [content], metadatas=[{"source": file_path.name, "type": "text"}]
                )
                documents.extend(docs)

            elif file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            qa_pair = json.loads(line.strip())
                            if "question" in qa_pair and "answer" in qa_pair:
                                self.qa_pairs.append(qa_pair)
                                combined_text = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
                                doc = Document(
                                    page_content=combined_text,
                                    metadata={
                                        "source": file_path.name,
                                        "type": "qa",
                                        "question": qa_pair["question"],
                                    },
                                )
                                documents.append(doc)
                        except json.JSONDecodeError:
                            continue

        if documents:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.config.collection_name,
                persist_directory=str(self.persist_directory),
            )
        else:
            # Create an empty collection if no documents found
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )

        self._save_qa_pairs()

    def _save_qa_pairs(self) -> None:
        # Ensure the persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        qa_pairs_file = self.persist_directory / "qa_pairs.pkl"
        with open(qa_pairs_file, "wb") as f:
            pickle.dump(self.qa_pairs, f)

    def _load_qa_pairs(self) -> None:
        qa_pairs_file = self.persist_directory / "qa_pairs.pkl"
        if qa_pairs_file.exists():
            with open(qa_pairs_file, "rb") as f:
                self.qa_pairs = pickle.load(f)

    def search(self, query: str) -> List[Document]:
        if not self.vectorstore:
            return []

        if self.config.use_mmr:
            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.config.top_k,
                fetch_k=self.config.mmr_fetch_k,
                lambda_mult=self.config.mmr_lambda
            )
        else:
            return self.vectorstore.similarity_search(query, k=self.config.top_k)

    def search_mmr(self, query: str, lambda_mult: float = None) -> List[Document]:
        """Search using MMR regardless of config setting"""
        if not self.vectorstore:
            return []

        lambda_val = lambda_mult if lambda_mult is not None else self.config.mmr_lambda
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.config.top_k,
            fetch_k=self.config.mmr_fetch_k,
            lambda_mult=lambda_val
        )

    def search_similarity(self, query: str) -> List[Document]:
        """Search using basic similarity regardless of config setting"""
        if not self.vectorstore:
            return []

        return self.vectorstore.similarity_search(query, k=self.config.top_k)

    def get_context(self, query: str) -> str:
        documents = self.search(query)
        context_parts = []

        for doc in documents:
            if doc.metadata.get("type") == "qa":
                context_parts.append(doc.page_content)
            else:
                context_parts.append(
                    f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
                )

        return "\n\n---\n\n".join(context_parts)
