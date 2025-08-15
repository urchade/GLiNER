
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>

#include "trie.h"

namespace cpp_trie {
    Node::Node(int k, bool p)
    {
        key = k;
        permanent = p;
    }

    int Node::get_key()
    {
        return key;
    }

    bool Node::is_permanent()
    {
        return permanent;
    }

    void Node::add_child(Node* child)
    {
        children.push_back(child);
    }

    Node* Node::get_child(int child_key)
    {
        for (Node* child : children) {
            if (child->key == child_key)
                return child;
        }
        return nullptr;
    }

    std::vector<Node*>& Node::get_children()
    {
        return children;
    }


    bool Node::has_children()
    {
        return !children.empty();
    }

    std::vector<cpp_trie::Node *>::iterator Node::find(int child_key)
    {
        for (auto i = children.begin(); i != children.end(); i++) {
            if ((*i)->get_key() == child_key)
                return i;
        }
        return children.end();
    }

    void Node::delete_child(int child_key)
    {   
        if (auto i = find(child_key); i != children.end())
            children.erase(i);
    }


    Node::~Node()
    {
        for (Node* child : children) {
            delete child;
        }
    }


    Trie::Trie()
    {   
        root = new Node(0, true);
    }

    Trie::Trie(std::vector<std::vector<int>>* init_value)
    {   
        root = new Node(0, true);
        if (init_value != nullptr)
            add_batch(init_value, true);
    }

    void Trie::add_batch(std::vector<std::vector<int>>* entities, bool permanent)
    {   
        for (std::vector<int> entity : *entities) {
            add(entity, permanent);
            entity.clear();
            entity.shrink_to_fit();
        }
    }

    void Trie::add(std::vector<int>& entity, bool permanent)
    {
        Node* current = root;
        for (int token_id : entity) {
            if (Node* next = current->get_child(token_id); next)
                current = next;
            else {
                Node* new_child = new Node(token_id, permanent);
                current->add_child(new_child);
                current = new_child;
            }
        }
    }

    std::vector<int> Trie::get_possible_next_keys(std::vector<int>& entity)
    {   
        std::vector<int> res;
        Node* tmp = root;
        for (int i : entity){
            if (Node* next = tmp->get_child(i); next) {
                tmp = next;
            } else {
                return res;
            }
        }

        std::vector<Node*> children = tmp->get_children();
        res.reserve(size(children));
        for (Node* c : children) {
            res.push_back(c->get_key());
        }
        return res;
    }

    std::vector<Node*> Trie::get_branch(std::vector<int>& entity)
    {   
        std::vector<Node*> b {root};
        Node* tmp = root;
        for (int token_id : entity) {
            if (Node* next = tmp->get_child(token_id); next)
                tmp = next;
            else
                return std::vector<Node*>{};

            b.push_back(tmp);
        }
        return b;
    }

    void Trie::remove_batch(std::vector<std::vector<int>>* entities)
    {   
        for (std::vector<int> entity : *entities) {
            remove_entity(entity);
        }
    }

    void Trie::remove_entity(std::vector<int>& entity)
    {
        std::vector<Node*> b = get_branch(entity);
        if (b.size() <= 1) {
            return;
        }

        for (auto c = b.rbegin(), p = b.rbegin()+1; p != b.rend(); c++, p++) {
            if ((*c)->has_children() || (*c)->is_permanent())
                break;
            (*p)->delete_child((*c)->get_key());
        }
    }

    Trie::~Trie()
    {
        delete root;
    }
};