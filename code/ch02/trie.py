

## 节点类 
class Node():
    def __init__(self) -> None:
        self.children = {}
        self.value = None
    
    # 增加节点
    def add_child(self, char, value, overwrite=False):
        child = self.children.get(char)
        if child is None:
            child = Node()                # 创建子节点
            self.children[char] = child   # 子节点赋值，字 -> 节点的映射
        
        if value is not None or overwrite:
            child.value = value           # 节点上对应的词
        
        return child
    
## 字典树  继承节点类   
class Trie(Node):

    def __contains__(self, key):
        return self[key] is not None
    
    # 查询方法
    def __getitem__(self, key):
        state = self
        for char in key:
            state = state.children.get(char)
            if state is None:
                return None
        
        return state.value
    
    # 重载方法，使得类可以像对待dict那样操作字典树
    # 构建一个词的字典树
    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key):
            if i < len(key) - 1:
                state = state.add_child(char, None)
            else:
                state = state.add_child(char, value, True)


if __name__ == '__main__':
    trie = Trie()
    # 增
    trie['自然'] = 'nature'
    trie['自然人'] = 'human'
    trie['自然语言'] = 'language'
    trie['自语'] = 'talk	to oneself'
    trie['入门'] = 'introduction'
    assert '自然' in trie
    # 删
    trie['自然'] = None
    assert '自然' not in trie
    # 改
    trie['自然语言'] = 'human language'
    assert trie['自然语言'] == 'human language'
    # 查
    assert trie['入门'] == 'introduction'
    print()
