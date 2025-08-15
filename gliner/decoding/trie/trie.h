#ifndef TRIE_H
#define TRIE_H

#include <vector>

namespace cpp_trie {
    class Node
    {
        int key;
        bool permanent;
        std::vector<Node*> children;

    public:
        Node(int k, bool permanent);
        int get_key();
        bool is_permanent();
        void add_child(Node* child);
        Node* get_child(int child_key);
        std::vector<Node*>& get_children();
        bool has_children();
        std::vector<cpp_trie::Node *>::iterator find(int child_key);
        void delete_child(int child_key);
        ~Node();
    };
    
    class Trie {
        Node* root;

        std::vector<Node*> get_branch(std::vector<int>& entity);
    public:
        Trie();
        Trie(std::vector<std::vector<int>>* init_value);
        void add_batch(std::vector<std::vector<int>>* entities, bool permanent);
        void add(std::vector<int>& entity, bool permanent);
        std::vector<int> get_possible_next_keys(std::vector<int>& entity);
        void remove_batch(std::vector<std::vector<int>>* entities);
        void remove_entity(std::vector<int>& entity);
        ~Trie();
    };
}

#endif