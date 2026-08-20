import os
import pyarrow.parquet as pq
from typing import List, Dict, Any, Generator, Callable
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

class IndicRAGChunker:

    # Devanagari & Bengali (hi, mr, ne, sa, as, bn)
    BRAHMIC_REGEX = "[^,।॥?!]+[,।॥?!]?"
    # Urdu (ur)
    URDU_REGEX = "[^،؛,۔؟!]+[،؛,۔؟!]?"
    # Latin conventions (en, ta, te, kn, ml, gu, pa)
    LATIN_REGEX = "[^,.;?!]+[,.;?!]?"

    def __init__(self, lang_code: str, chunk_size_limit: int = 2000, tokenizer: Callable[[str], list] = list):

        self.lang_code = lang_code
        self.chunk_size_limit = chunk_size_limit
        self.tokenizer = tokenizer
        
        # Route the language to the correct script family regex
        if lang_code in ["hi", "mr", "ne", "sa", "as", "bn"]:
            chunking_regex = self.BRAHMIC_REGEX
        elif lang_code == "ur":
            chunking_regex = self.URDU_REGEX
        else:
            chunking_regex = self.LATIN_REGEX
            
        # LlamaIndex SentenceSplitter
        self.safety_splitter = SentenceSplitter(
            chunk_size=self.chunk_size_limit,
            chunk_overlap=50,  # Small overlap only triggered if a split is forced
            secondary_chunking_regex=chunking_regex,
            # We override the tokenizer to not use Tiktoken for Indic languages
            tokenizer=self.tokenizer 
        )

    def process_record(self, record: Dict[str, Any]) -> List[LlamaDocument]:
        """
        Processes a single structured Parquet row.
        Returns a list of LlamaIndex Documents.
        """
        nodes = []
        
        query_id = record.get("query_id")
        
        # PyArrow nested structs parsing
        passages_dict = record.get("passages", {})
        try:
            is_selected_list = passages_dict.get("is_selected", [])
            eng_passages = passages_dict.get("English_passages", [])
            translated_passages = passages_dict.get("Translated_passages", [])
        except AttributeError:
            return nodes
            
        for idx in range(len(is_selected_list)):
            is_selected = is_selected_list[idx]
            eng_text = eng_passages[idx] if idx < len(eng_passages) else ""
            translated_text = translated_passages[idx] if idx < len(translated_passages) else ""
            
            # INDEX ENGLISH PASSAGE
            if eng_text:
                # English uses Latin regex explicitly
                eng_splitter = SentenceSplitter(
                    chunk_size=self.chunk_size_limit, 
                    chunk_overlap=50,
                    secondary_chunking_regex=self.LATIN_REGEX,
                    tokenizer=self.tokenizer
                )
                eng_fragments = eng_splitter.split_text(eng_text)
                
                for frag_idx, frag_text in enumerate(eng_fragments):
                    nodes.append(LlamaDocument(
                        text=frag_text,
                        metadata={
                            "query_id": str(query_id),
                            "passage_idx": idx,
                            "is_selected": bool(is_selected),
                            "lang": "en",
                            "is_translated_corpus": False,
                            "is_fragment": len(eng_fragments) > 1,
                            "fragment_idx": frag_idx
                        }
                    ))
            
            # INDEX TRANSLATED PASSAGE
            if translated_text:
                translated_fragments = self.safety_splitter.split_text(translated_text)
                
                for frag_idx, frag_text in enumerate(translated_fragments):
                    nodes.append(LlamaDocument(
                        text=frag_text,
                        metadata={
                            "query_id": str(query_id),
                            "passage_idx": idx,
                            "is_selected": bool(is_selected),
                            "lang": self.lang_code,
                            "is_translated_corpus": True,
                            "is_fragment": len(translated_fragments) > 1,
                            "fragment_idx": frag_idx
                        }
                    ))
                    
        return nodes

def stream_and_chunk_parquet(file_path: str, lang_code: str, chunk_size_limit: int = 2000, batch_size: int = 100) -> Generator[List[LlamaDocument], None, None]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    chunker = IndicRAGChunker(lang_code=lang_code, chunk_size_limit=chunk_size_limit)
    pf = pq.ParquetFile(file_path)
    
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        batch_nodes = []
        
        for _, row in df.iterrows():
            record = row.to_dict()
            nodes = chunker.process_record(record)
            batch_nodes.extend(nodes)
            
        yield batch_nodes

if __name__ == "__main__":
    test_file = "data/train/hintrain.parquet"
    if os.path.exists(test_file):
        print(f"Processing first batch from {test_file} for language 'hi'...")
        for batch_nodes in stream_and_chunk_parquet(test_file, lang_code="hi", chunk_size_limit=2000, batch_size=2):
            print(f"Extracted {len(batch_nodes)} LlamaIndex nodes.")
            hi_node = next((n for n in batch_nodes if n.metadata['lang'] == 'hi'), None)
            if hi_node:
                print(hi_node.text)
                print(f"Metadata: {hi_node.metadata}")
            break
