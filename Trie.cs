//208. ÊµÏÖ Trie (Ç°×ºÊ÷)
//https://leetcode-cn.com/problems/implement-trie-prefix-tree/
public class Trie
{
    class TrieNode
    {
        public bool IsWord;
        public TrieNode[] childs = new TrieNode[26];
    }
    TrieNode[] nodes = new TrieNode[26];
    /** Initialize your data structure here. */
    public Trie()
    {

    }

    /** Inserts a word into the trie. */
    public void Insert(string word)
    {
        var setNodes = nodes;
        for (int i = 0; i < word.Length; i++)
        {
            var ch = word[i] - 'a';
            if (setNodes[ch] == null)
            {
                setNodes[ch] = new TrieNode();
            }
            setNodes[ch].IsWord = setNodes[ch].IsWord || i == word.Length - 1;
            setNodes = setNodes[ch].childs;
        }
    }

    /** Returns if the word is in the trie. */
    public bool Search(string word)
    {
        var queryNodes = nodes;
        for (int i = 0; i < word.Length; i++)
        {
            var ch = word[i] - 'a';
            if (queryNodes[ch] == null)
            {
                return false;
            }
            if (i == word.Length - 1 && queryNodes[ch].IsWord)
            {
                return true;
            }
            queryNodes = queryNodes[ch].childs;
        }
        return false;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public bool StartsWith(string prefix)
    {
        var queryNodes = nodes;
        for (int i = 0; i < prefix.Length; i++)
        {
            var ch = prefix[i] - 'a';
            if (queryNodes[ch] == null)
            {
                return false;
            }
            queryNodes = queryNodes[ch].childs;
        }
        return true;
    }
}