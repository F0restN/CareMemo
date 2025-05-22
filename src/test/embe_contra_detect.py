# NOTE: Added package: sentence-transformers, torch, einops



import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class ContradictionDetector:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize with either Nomic or BGE model
        Options:
        - "nomic-ai/nomic-embed-text-v1.5" (Nomic)
        - "BAAI/bge-large-en-v1.5" (BGE)
        - "BAAI/bge-base-en-v1.5" (BGE smaller)
        """
        # Add trust_remote_code=True for Nomic model
        if "nomic" in model_name.lower():
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        
    def get_embeddings(self, sentences, task_type="search_query"):
        """Get embeddings with task-specific prefixes for better performance"""
        
        # Add task prefixes for better performance (especially for BGE)
        if "bge" in self.model_name.lower():
            # BGE models benefit from instruction prefixes
            prefixed_sentences = [f"Represent this sentence: {sent}" for sent in sentences]
        elif "nomic" in self.model_name.lower():
            # Nomic models can use task-specific prefixes
            prefixed_sentences = [f"search_query: {sent}" for sent in sentences]
        else:
            prefixed_sentences = sentences
            
        embeddings = self.model.encode(prefixed_sentences, normalize_embeddings=True)
        return embeddings
    
    def detect_contradiction_cosine(self, sentence1, sentence2, threshold=0.8):
        """
        Detect contradiction using cosine similarity
        Lower similarity often indicates contradiction
        """
        embeddings = self.get_embeddings([sentence1, sentence2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Low similarity might indicate contradiction
        is_contradiction = similarity < threshold
        
        return {
            'is_contradiction': is_contradiction,
            'similarity_score': float(similarity),
            'confidence': 1 - similarity if is_contradiction else similarity
        }
    
    def detect_contradiction_contrast(self, sentence1, sentence2):
        """
        Advanced method: Use contrastive embeddings
        Create positive and negative versions to better detect contradictions
        """
        # Original sentences
        original_emb = self.get_embeddings([sentence1, sentence2])
        
        # Create explicit contradiction versions
        neg_sentence1 = f"It is not true that {sentence1.lower()}"
        neg_sentence2 = f"It is not true that {sentence2.lower()}"
        
        negative_emb = self.get_embeddings([neg_sentence1, neg_sentence2])
        
        # Calculate similarities
        original_sim = cosine_similarity([original_emb[0]], [original_emb[1]])[0][0]
        cross_sim1 = cosine_similarity([original_emb[0]], [negative_emb[1]])[0][0]
        cross_sim2 = cosine_similarity([original_emb[1]], [negative_emb[0]])[0][0]
        
        # If sentence1 is more similar to negation of sentence2, they likely contradict
        contradiction_score = max(cross_sim1, cross_sim2)
        
        is_contradiction = contradiction_score > original_sim
        
        return {
            'is_contradiction': is_contradiction,
            'original_similarity': float(original_sim),
            'contradiction_score': float(contradiction_score),
            'confidence': float(abs(contradiction_score - original_sim))
        }
    
    def batch_detect_contradictions(self, sentence_pairs, method="cosine"):
        """Detect contradictions for multiple sentence pairs"""
        results = []
        
        for pair in sentence_pairs:
            if method == "cosine":
                result = self.detect_contradiction_cosine(pair[0], pair[1])
            else:
                result = self.detect_contradiction_contrast(pair[0], pair[1])
            
            results.append({
                'sentence1': pair[0],
                'sentence2': pair[1],
                **result
            })
        
        return results

# Example usage
def main():
    # Initialize with different models
    print("Testing with Nomic model...")
    detector_nomic = ContradictionDetector("nomic-ai/nomic-embed-text-v1.5")
    
    print("Testing with BGE model...")
    detector_bge = ContradictionDetector("BAAI/bge-large-en-v1.5")
    
    # Test cases
    test_pairs = [
        ("The weather is sunny today", "It's raining heavily outside"),
        ("The cat is sleeping on the couch", "The cat is playing in the garden"),
        ("John is 25 years old", "John is a young adult"),
        ("The store is open", "The store is closed"),
        ("She loves chocolate", "She enjoys sweet treats")
    ]
    
    print("\n" + "="*60)
    print("NOMIC MODEL RESULTS")
    print("="*60)
    
    for pair in test_pairs:
        # Test cosine method
        result_cosine = detector_nomic.detect_contradiction_cosine(pair[0], pair[1])
        print(f"\nSentence 1: {pair[0]}")
        print(f"Sentence 2: {pair[1]}")
        print(f"Cosine Method - Contradiction: {result_cosine['is_contradiction']}")
        print(f"Similarity: {result_cosine['similarity_score']:.3f}")
        
        # Test contrast method
        result_contrast = detector_nomic.detect_contradiction_contrast(pair[0], pair[1])
        print(f"Contrast Method - Contradiction: {result_contrast['is_contradiction']}")
        print(f"Contradiction Score: {result_contrast['contradiction_score']:.3f}")
    
    print("\n" + "="*60)
    print("BGE MODEL RESULTS")
    print("="*60)
    
    for pair in test_pairs:
        result_cosine = detector_bge.detect_contradiction_cosine(pair[0], pair[1])
        print(f"\nSentence 1: {pair[0]}")
        print(f"Sentence 2: {pair[1]}")
        print(f"Cosine Method - Contradiction: {result_cosine['is_contradiction']}")
        print(f"Similarity: {result_cosine['similarity_score']:.3f}")

if __name__ == "__main__":
    main()

# Additional utility functions
def compare_models_performance():
    """Compare different embedding models on contradiction detection"""
    models = [
        "nomic-ai/nomic-embed-text-v1.5",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5"
    ]
    
    test_cases = [
        ("The sky is blue", "The sky is red", True),
        ("He is tall", "He is short", True),
        ("The book is interesting", "The book is engaging", False),
        ("It's hot outside", "It's cold outside", True),
        ("She is happy", "She is joyful", False),
        ("My job is a software engineer", "I work as a constractor", True),
        ("I'm taking care of my dad", "I take care of my mom", True),
        ("I'm a vegetarian", "I don't eat protein or meat", False),
    ]
    
    for model_name in models:
        print(f"\n{model_name}:")
        detector = ContradictionDetector(model_name)
        
        correct = 0
        for sent1, sent2, expected_contradiction in test_cases:
            result = detector.detect_contradiction_cosine(sent1, sent2)
            predicted = result['is_contradiction']
            
            if predicted == expected_contradiction:
                correct += 1
                
            print(f"  {sent1} | {sent2}")
            print(f"  Expected: {expected_contradiction}, Got: {predicted}, Sim: {result['similarity_score']:.3f}")
        
        accuracy = correct / len(test_cases)
        print(f"  Accuracy: {accuracy:.2%}")

# Uncomment to compare models
compare_models_performance()