import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the ML models used in the application."""
    embeddings_model: str = "text-embedding-3-small"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_persist_dir: str = "./chroma_db"
    retriever_k: int = 5
    rerank_k: int = 3
    max_token_length: int = 512

class JobMatcher:
    """Main class for handling job matching functionality."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialize_environment()
        self._initialize_models()
    
    def _initialize_environment(self) -> None:
        """Initialize environment variables and configurations."""
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            os.environ["OPENAI_API_KEY"] = api_key
        except Exception as e:
            logger.error(f"Error initializing environment: {str(e)}")
            raise

    def _initialize_models(self) -> None:
        """Initialize the embedding and cross-encoder models."""
        try:
            self.embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
            self.vectorstore = Chroma(
                persist_directory=self.config.chroma_persist_dir,
                embedding_function=self.embeddings
            )
            if self.vectorstore._collection.count() == 0:
                with st.spinner("Loading documents for knowledge base..."):
                    docs = pd.read_csv('allJobs.csv/allJobs.csv', nrows=500)
                    docs['id'] = docs.index
                    docs['full_description'] = docs.apply(lambda row: ' | '.join([f'{docs.columns[i]}: {row[i]}' for i in range(len(row))]), axis=1)
                    docs.drop_duplicates(subset=['full_description'], inplace=True)
                    self.vectorstore.add_texts(docs['full_description'].values, embedding=self.embeddings, metadatas=docs.drop(columns=['id', 'full_description']).to_dict('records')) 

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever_k}
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.cross_encoder_model)
            self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(
                self.config.cross_encoder_model
            )
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def compute_relevance_score(self, query: str, context: str) -> float:
        """Compute relevance score between query and context using cross-encoder."""
        try:
            inputs = self.tokenizer(
                query,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_token_length
            )
            with torch.no_grad():
                outputs = self.cross_encoder(**inputs)
            return outputs.logits.item()
        except Exception as e:
            logger.error(f"Error computing relevance score: {str(e)}")
            return 0.0

    def rerank_contexts(self, query: str, contexts: List[Any]) -> List[Any]:
        """Rerank contexts based on relevance scores."""
        try:
            scored_contexts = [
                (context, self.compute_relevance_score(query, context.page_content))
                for context in contexts
            ]
            return [
                context for context, _ in sorted(
                    scored_contexts,
                    key=lambda x: x[1],
                    reverse=True
                )[:self.config.rerank_k]
            ]
        except Exception as e:
            logger.error(f"Error reranking contexts: {str(e)}")
            return contexts[:self.config.rerank_k]

class JobMatcherUI:
    """Class for handling the Streamlit UI components."""
    
    def __init__(self, job_matcher: JobMatcher):
        self.job_matcher = job_matcher
        
    def render_metadata(self, metadata: Dict[str, Any], n_columns: int = 4) -> None:
        """Render job metadata in a structured format."""
        try:
            cols = st.columns(n_columns)
            sorted_keys = sorted(k for k in metadata.keys() if k != 'Description')
            
            for idx, key in enumerate(sorted_keys):
                with cols[idx % n_columns]:
                    st.markdown(f"**{key}:**")
                    value = metadata[key]
                    
                    if isinstance(value, list):
                        st.write(", ".join(map(str, value)))
                    elif isinstance(value, str) and len(value) > 50:
                        st.write(
                            f'<div style="overflow-wrap: break-word;">{value}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(value)
            
            if 'Description' in metadata:
                with st.expander("Job Description", expanded=False):
                    st.write(metadata['Description'])
                    
            st.markdown("---")
        except Exception as e:
            logger.error(f"Error rendering metadata: {str(e)}")
            st.error("Error displaying job details")

    def process_cv(self, cv_file) -> None:
        """Process uploaded CV and display matching jobs."""
        try:
            cv = PyPDFLoader(cv_file.name).load()[0]
            contexts = self.job_matcher.retriever.invoke(cv.page_content)
            reranked_contexts = self.job_matcher.rerank_contexts(cv.page_content, contexts)
            
            st.subheader("Matching Jobs")
            for i, context in enumerate(reranked_contexts, 1):
                with st.container():
                    st.write(f"Match #{i}")
                    self.render_metadata(context.metadata)
                    
        except Exception as e:
            logger.error(f"Error processing CV: {str(e)}")
            st.error("Error processing CV. Please try again.")

def main():
    """Main application entry point."""
    try:
        st.set_page_config(
            page_title="AI Job Matcher",
            page_icon="💼",
            layout="wide"
        )
        
        st.title("AI Job Matching")
        st.markdown("""
        Upload your CV to find matching job opportunities.
        Supported format: PDF
        """)
        
        config = ModelConfig()
        job_matcher = JobMatcher(config)
        ui = JobMatcherUI(job_matcher)
        
        cv_file = st.file_uploader("Upload your CV", type="pdf")
        
        if cv_file:
            if st.button("Find Matching Jobs", type="primary"):
                with st.spinner("Analyzing your CV and finding matches..."):
                    ui.process_cv(cv_file)
                    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()

