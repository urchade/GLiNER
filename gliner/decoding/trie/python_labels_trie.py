from typing import List, Optional

class Node:
    __slots__ = ("_key", "_permanent", "_children")

    def __init__(self, key: int, permanent: bool):
        self._key = key
        self._permanent = permanent
        self._children: dict[int, "Node"] = {}

    def get_key(self) -> int:
        return self._key

    def is_permanent(self) -> bool:
        return self._permanent

    def add_child(self, child: "Node") -> None:
        self._children[child.get_key()] = child

    def get_child(self, child_key: int) -> Optional["Node"]:
        return self._children.get(child_key)

    def get_children(self) -> List["Node"]:
        # Preserve insertion order like iterating a vector
        return list(self._children.values())

    def has_children(self) -> bool:
        return bool(self._children)

    def delete_child(self, child_key: int) -> None:
        self._children.pop(child_key, None)


class Trie:
    def __init__(self, init_value: Optional[List[List[int]]] = None):
        # Root has key=0 and is permanent (matches the C++ code)
        self.root = Node(0, True)
        if init_value:
            self.add_batch(init_value, permanent=True)

    def add_batch(self, entities: List[List[int]], permanent: bool) -> None:
        for entity in entities:
            self.add(entity, permanent)

    def add(self, entity: List[int], permanent: bool) -> None:
        current = self.root
        for token_id in entity:
            nxt = current.get_child(token_id)
            if nxt is None:
                nxt = Node(token_id, permanent)
                current.add_child(nxt)
            current = nxt

    def get_possible_next_keys(self, entity: List[int]) -> List[int]:
        tmp = self.root
        for token_id in entity:
            nxt = tmp.get_child(token_id)
            if nxt is None:
                return []
            tmp = nxt
        return [child.get_key() for child in tmp.get_children()]

    def get_branch(self, entity: List[int]) -> List[Node]:
        # Includes root at position 0 when the full path exists.
        branch = [self.root]
        tmp = self.root
        for token_id in entity:
            nxt = tmp.get_child(token_id)
            if nxt is None:
                return []
            tmp = nxt
            branch.append(tmp)
        return branch

    def remove_batch(self, entities: List[List[int]]) -> None:
        for entity in entities:
            self.remove_entity(entity)

    def remove_entity(self, entity: List[int]) -> None:
        branch = self.get_branch(entity)
        # If not found or only root, nothing to remove
        if len(branch) <= 1:
            return
        for child, parent in zip(reversed(branch[1:]), reversed(branch[:-1])):
            if child.has_children() or child.is_permanent():
                break
            parent.delete_child(child.get_key())

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