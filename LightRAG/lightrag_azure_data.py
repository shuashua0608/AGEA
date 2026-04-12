import os
import asyncio
import json
import argparse
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
AGEA_ROOT = SCRIPT_DIR / "AGEA_src" if (SCRIPT_DIR / "AGEA_src").exists() else SCRIPT_DIR
DATASET_INPUT_ROOT = AGEA_ROOT
GRAPH_OUTPUT_ROOT = AGEA_ROOT / "graphs"

env_file = AGEA_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")


def get_available_datasets():
    """Get list of available datasets under AGEA input root."""
    available_datasets = []
    
    # Define the datasets we want to process
    target_datasets = ["medical", "novel", "novel_3", "novel_9", "novel_13", "novel_15", "agriculture"]
    
    # Check which datasets exist under AGEA input root
    for dataset_name in target_datasets:
        dataset_path = DATASET_INPUT_ROOT / dataset_name
        corpus_path = os.path.join(dataset_path, "Corpus.json")
        if os.path.exists(corpus_path):
            available_datasets.append(dataset_name)
        else:
            print(f"Warning: Corpus.json not found for dataset: {dataset_name} at {corpus_path}")
    
    return sorted(available_datasets)


def load_corpus_data(dataset_name):
    """
    Load corpus data from the specified dataset.
    Handles multiple formats:
    - Medical format: [{"corpus_name": "Medical", "context": "...", ...}]
    - Novel format: [{"idx": ..., "title": "...", "text": "...", ...}]
    - Agriculture format: {"context": "..."} separated by newlines
    - Plain text format: raw text content
    """
    corpus_path = DATASET_INPUT_ROOT / dataset_name / "Corpus.json"
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus.json not found for dataset: {dataset_name} at {corpus_path}")
    
    print(f"Loading corpus from: {corpus_path}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    contexts = []
    
    # Try to parse as JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            # Check for different formats in the list
            for item in data:
                if isinstance(item, dict):
                    # Medical format: has 'context' key
                    if 'context' in item:
                        contexts.append(item['context'])
                    # Novel format: has 'text' key
                    elif 'text' in item:
                        contexts.append(item['text'])
                    # Fallback: if dict has string values, try to use the longest one
                    else:
                        # Try to find a text-like field
                        text_fields = [v for k, v in item.items() if isinstance(v, str) and len(v) > 100]
                        if text_fields:
                            contexts.append(max(text_fields, key=len))
            
            if contexts:
                print(f"Loaded {len(contexts)} contexts from JSON array format")
                return contexts
    except json.JSONDecodeError:
        pass
    
    # Try to parse as separate JSON objects (Agriculture format)
    try:
        # Split by }{ to separate JSON objects
        parts = content.split('}{')
        
        for i, part in enumerate(parts):
            if i == 0:
                part = part + '}'
            elif i == len(parts) - 1:
                part = '{' + part
            else:
                part = '{' + part + '}'
            
            try:
                json_obj = json.loads(part)
                if isinstance(json_obj, dict):
                    if 'context' in json_obj:
                        contexts.append(json_obj['context'])
                    elif 'text' in json_obj:
                        contexts.append(json_obj['text'])
            except json.JSONDecodeError:
                continue
        
        if contexts:
            print(f"Loaded {len(contexts)} contexts from separate JSON objects format")
            return contexts
    except Exception as e:
        print(f"Error parsing separate JSON objects: {e}")
    
    # If all else fails, try to treat the entire content as one context
    if content:
        contexts.append(content)
        print(f"Loaded 1 context from raw text format")
        return contexts
    
    raise ValueError(f"Could not parse corpus data from {corpus_path}")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)


async def get_embedding_dimension() -> int:
    """
    Auto-detect embedding dimension by calling the embedding function with a test input.
    This ensures the dimension matches the actual model being used.
    """
    try:
        test_result = await embedding_func(["test"])
        if len(test_result.shape) == 2:
            return int(test_result.shape[1])
        elif len(test_result.shape) == 1:
            return int(test_result.shape[0])
        else:
            raise ValueError(f"Unexpected embedding shape: {test_result.shape}")
    except Exception as e:
        # Fallback: try to determine from model name
        model_name = AZURE_EMBEDDING_DEPLOYMENT or ""
        if "large" in model_name.lower():
            return 3072
        elif "small" in model_name.lower():
            return 1536
        else:
            # Default fallback
            print(f"Warning: Could not auto-detect embedding dimension, defaulting to 1536. Error: {e}")
            return 1536


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("LLM Response: ", result)

    result = await embedding_func(["How are you?"])
    print("Embedding Result: ", result.shape)
    print("Embedding Dimension: ", result.shape[1])


async def initialize_rag(working_dir, chunk_token_size=1200, chunk_overlap_token_size=100):
    # Auto-detect embedding dimension to match the actual model
    embedding_dim = await get_embedding_dimension()
    print(f"Detected embedding dimension: {embedding_dim} (from model: {AZURE_EMBEDDING_DEPLOYMENT})")
    
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    parser = argparse.ArgumentParser(description='LightRAG Azure Data Processing')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (medical, novel, novel_3, novel_9, novel_13, or novel_15)')
    parser.add_argument('--test-llm', action='store_true',
                       help='Test LLM and embedding functions before processing')
    parser.add_argument('--query', type=str, default="Who are the main characters?",
                       help='Query to test the knowledge graph')
    parser.add_argument('--top-k', type=int, default=30,
                       help='Number of top entities/relations to retrieve (default: 30)')
    parser.add_argument('--chunk-top-k', type=int, default=30,
                       help='Number of text chunks to retrieve (default: 30)')
    parser.add_argument('--max-entity-tokens', type=int, default=4000,
                       help='Maximum tokens for entity context (default: 4000)')
    parser.add_argument('--max-relation-tokens', type=int, default=4000,
                       help='Maximum tokens for relation context (default: 4000)')
    parser.add_argument('--max-total-tokens', type=int, default=8000,
                       help='Maximum total tokens for query context (default: 8000)')
    parser.add_argument('--chunk-token-size', type=int, default=1200,
                       help='Maximum tokens per text chunk when splitting documents (default: 1200)')
    parser.add_argument('--chunk-overlap-token-size', type=int, default=100,
                       help='Overlapping tokens between consecutive chunks (default: 100)')
    
    args = parser.parse_args()
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    if args.dataset not in available_datasets:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {', '.join(available_datasets)}")
        return
    
    print(f"Processing dataset: {args.dataset}")
    
    # Test LLM functions if requested
    if args.test_llm:
        print("Testing LLM and embedding functions...")
        asyncio.run(test_funcs())
        print("LLM test completed.\n")
    
    # Set up working directory - save to AGEA_src/graphs/{dataset}
    datasets_dir = GRAPH_OUTPUT_ROOT
    working_dir = datasets_dir / args.dataset
    
    # Create datasets directory if it doesn't exist
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)
        print(f"Created datasets directory: {datasets_dir}")
    
    if os.path.exists(working_dir):
        import shutil
        print(f"Removing existing directory: {working_dir}")
        shutil.rmtree(working_dir)
    
    os.makedirs(working_dir, exist_ok=True)
    print(f"Created working directory: {working_dir}")
    
    # Load corpus data
    try:
        contexts = load_corpus_data(args.dataset)
        print(f"Successfully loaded {len(contexts)} contexts")
    except Exception as e:
        print(f"Error loading corpus data: {e}")
        return
    
    # Initialize and run RAG
    print("Initializing LightRAG...")
    print(f"Chunking Parameters:")
    print(f"  chunk_token_size: {args.chunk_token_size}")
    print(f"  chunk_overlap_token_size: {args.chunk_overlap_token_size}")
    rag = asyncio.run(initialize_rag(
        working_dir,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size
    ))
    
    print("Inserting contexts into knowledge graph...")
    rag.insert(contexts)
    
    print("Knowledge graph construction completed!")
    print(f"Results saved in: {working_dir}")
    
    # Run test queries
    print(f"\nTesting query: '{args.query}'")
    
    # Create answers directory if it doesn't exist - save to A_TAG/datasets/{dataset}/answers
    answers_dir = os.path.join(working_dir, "answers")
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir, exist_ok=True)
        print(f"Created answers directory: {answers_dir}")
    
    # Generate answers for all modes
    modes = ["naive", "local", "global", "hybrid"]
    answers = {}
    
    # Configure query parameters (will be used for all modes)
    base_query_params = {
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "max_entity_tokens": args.max_entity_tokens,
        "max_relation_tokens": args.max_relation_tokens,
        "max_total_tokens": args.max_total_tokens
    }
    
    print(f"\nQuery Parameters:")
    print(f"  top_k: {base_query_params['top_k']}")
    print(f"  chunk_top_k: {base_query_params['chunk_top_k']}")
    print(f"  max_entity_tokens: {base_query_params['max_entity_tokens']}")
    print(f"  max_relation_tokens: {base_query_params['max_relation_tokens']}")
    print(f"  max_total_tokens: {base_query_params['max_total_tokens']}")
    
    for mode in modes:
        print(f"\nResult ({mode.title()}):")
        # Create QueryParam with mode and other parameters
        query_param = QueryParam(
            mode=mode,
            **base_query_params
        )
        answer = rag.query(args.query, param=query_param)
        answers[mode] = answer
        print(answer)
    
    # Save query and answers to file
    answer_file = os.path.join(answers_dir, f"{args.dataset}.txt")
    with open(answer_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Query: {args.query}\n")
        f.write("=" * 80 + "\n\n")
        
        for mode, answer in answers.items():
            f.write(f"Answer ({mode.title()} Mode):\n")
            f.write("-" * 40 + "\n")
            f.write(f"{answer}\n\n")
    
    print(f"\nQuery and answers saved to: {answer_file}")


if __name__ == "__main__":
    main()
