from typing import List
from .trie import Trie

class LabelsTrie:
    def __init__(self, entities: List[List[int]] = None):
        """Initialize the trie.
        
        Args:
            entities: Optional initial list of token sequences to add to the trie.
                     If None or empty, creates an empty trie.
        """
        if not entities:
            self.trie = Trie()
        else:
            self.trie = Trie(entities)

    def add_batch(self, entities: List[List[int]]):
        """Add multiple token sequences to the trie.
        
        Args:
            entities: List of token sequences to add.
        """
        self.trie.add_batch(entities, permanent=False)

    def add(self, tokens: List[int]):
        """Add a single token sequence to the trie.
        
        Args:
            tokens: Token sequence to add.
        """
        self.trie.add(tokens, permanent=False)

    def get(self, prefix: List[int]) -> List[int]:
        """Get possible next tokens after a given prefix.
        
        Args:
            prefix: The token sequence to search for.
            
        Returns:
            List of possible next token IDs.
        """
        return self.trie.get_possible_next_keys(prefix)

    def get_root(self) -> List[int]:
        """Get all direct children keys of the root node.
        
        Returns:
            List of token IDs that are direct children of the root.
        """
        return [child.key for child in self.trie.root.children]
    
    def remove_batch(self, entities: List[List[int]]):
        """Remove multiple token sequences from the trie.
        
        Args:
            entities: List of token sequences to remove.
        """
        self.trie.remove_batch(entities)

    def remove_entity(self, tokens: List[int]):
        """Remove a single token sequence from the trie.
        
        Args:
            tokens: Token sequence to remove.
        """
        self.trie.remove_entity(tokens)