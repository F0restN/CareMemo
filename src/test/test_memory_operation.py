"""Simple test cases for MemoryOperation class using pytest with real vector store."""

import pytest
from datetime import datetime, timezone

from classes.Memory import MemoryItem, CategoryEnum
from classes.MemoryOperation import MemoryOperation


class TestMemoryOperationBasic:
    """Basic tests for MemoryOperation functionality."""

    @pytest.fixture
    def memory_operation(self):
        """Create MemoryOperation instance for testing."""
        # Collection name will be provided later by user
        return MemoryOperation(collection_name="test_collection")

    @pytest.fixture
    def sample_memory_item(self):
        """Create a sample MemoryItem for testing."""
        return MemoryItem(
            content="User prefers coffee in the morning",
            level="LTM",
            category=CategoryEnum.PREFERENCES,
            type="beverage preference",
            topic=["coffee", "morning", "drink"],
            user_id="test_user",
            source="test_conversation"
        )

    def test_memory_operation_initialization(self, memory_operation):
        """Test that MemoryOperation initializes correctly."""
        assert memory_operation is not None
        assert memory_operation.kb is not None
        assert memory_operation.collection_name == "test_collection"

    def test_add_memory_item(self, memory_operation, sample_memory_item):
        """Test adding a memory item to the vector store."""
        result = memory_operation.add_to_memory(sample_memory_item)
        assert result is not None
        print(f"Added memory with result: {result}")

    def test_recall_memory(self, memory_operation):
        """Test recalling memories from the vector store."""
        result = memory_operation.recall_memory(
            query="coffee preferences", 
            user_id="test_user",
            k=5
        )
        assert isinstance(result, list)
        print(f"Recalled {len(result)} memories")

    def test_update_memory(self, memory_operation, sample_memory_item):
        """Test updating a memory in the vector store."""
        
        memories = memory_operation.recall_memory(
            query="coffee preferences", 
            user_id="test_user",
            k=1
        )
        assert len(memories) == 1
        premise_memory = memories[0]
        
        result = memory_operation.update_memory(
            premise_memory=premise_memory,
            new_memory=sample_memory_item,
        )
        assert result is not None
        print(f"Updated memory with result: {result}")
        

class TestMemoryOperationAdvanced:
    """Advanced test scenarios for MemoryOperation."""

    @pytest.fixture
    def memory_operation(self):
        """Create MemoryOperation instance for testing."""
        return MemoryOperation(collection_name="test_collection")

    def test_multiple_memory_types(self, memory_operation):
        """Test adding different types of memories."""
        memories = [
            MemoryItem(
                content="User works as a software engineer",
                level="LTM",
                category=CategoryEnum.BIO_INFO,
                type="profession",
                topic=["job", "engineer", "software"],
                user_id="multi_test_user",
                source="profile_setup"
            ),
            MemoryItem(
                content="User has a father with Alzheimer's disease",
                level="LTM",
                category=CategoryEnum.ADRD_INFO,
                type="family medical history",
                topic=["father", "alzheimer", "family"],
                user_id="multi_test_user",
                source="medical_history"
            ),
            MemoryItem(
                content="User prefers gentle communication style",
                level="STM",
                category=CategoryEnum.PREFERENCES,
                type="communication preference",
                topic=["gentle", "communication", "style"],
                user_id="multi_test_user",
                source="interaction_feedback"
            )
        ]
        
        results = []
        for memory in memories:
            result = memory_operation.add_to_memory(memory)
            results.append(result)
            assert result is not None
        
        print(f"Added {len(results)} different memory types")

    def test_search_specificity(self, memory_operation):
        """Test search with different queries and parameters."""
        test_cases = [
            {"query": "software engineering", "user_id": "multi_test_user", "k": 3},
            {"query": "alzheimer family", "user_id": "multi_test_user", "k": 5},
            {"query": "communication preferences", "user_id": "multi_test_user", "score": 0.7, "k": 2}
        ]
        
        for test_case in test_cases:
            result = memory_operation.recall_memory(**test_case)
            assert isinstance(result, list)
            print(f"Query '{test_case['query']}' returned {len(result)} results")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
