//  Implement strStr() https://leetcode.com/problems/implement-strstr/

class Solution {
public:
    int strStr(string haystack, string needle) {
        int len1 = haystack.size();
        int len2 = needle.size();
        if (len2 == 0) return 0;
        for (int i = 0; i < len1 - len2; i++) {
            if (i == len1) return -1;
            int j = 0;
            for (;j < len2; j++) {
                if (haystack[i + j] != needle[j]) break;
            }
            if (j == len2) return i;
        }
        return -1;
    }
};

// Valid Anagram https://leetcode.com/problems/valid-anagram/

class Solution {
public:
    bool isAnagram(string s, string t) {
       if (s.size() != t.size()) return false;
       
       int table[26] = {0};
       
       for(int i = 0; i < s.size(); i++) {
           table[s[i] - 'a']++;
       }
       
       for (int j = 0; j < t.size(); j++) {
           table[t[j] - 'a']--;
           if(table[t[j] - 'a'] < 0) return false;
       }
       return true;
    }
};

// Group Anagrams(TLE) https://leetcode.com/problems/anagrams/

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string> > res;
        for (int i = 0; i < strs.size(); i++) {
            bool flag = false;
            for(int j = 0; j < res.size(); j++) {
                if(isAnagram(res[j][0], strs[i]) ) {res[j].push_back(strs[i]); flag = true; break;}
            }
            if (!flag) {
                vector<string> temp;
                temp.push_back(strs[i]);
                res.push_back(temp);
            }
        }
        return res;
    }
    
    bool isAnagram(string s, string t) {
       if (s.size() != t.size()) return false;
       
       int table[26] = {0};
       
       for(int i = 0; i < s.size(); i++) {
           table[s[i] - 'a']++;
       }
       
       for (int j = 0; j < t.size(); j++) {
           table[t[j] - 'a']--;
           if(table[t[j] - 'a'] < 0) return false;
       }
       return true;
    }
};

// Group Anagrams https://leetcode.com/problems/anagrams/

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        map<string , vector<int> > table;
        map<string, bool> flag;
        for (int i = 0; i < strs.size(); i++) {
            string s = strs[i];
            sort(s.begin(), s.end());
            table[s].push_back(i);
            flag[s] = false;
        }
        
       vector<vector<string> > res;
        
        for (int i = 0; i < strs.size(); i++) {
            string s = strs[i];
            sort(s.begin(), s.end());
            if (!flag[s]) {
                vector<string> temp;
                for (int j = 0; j < table[s].size(); j++) {
                    temp.push_back(strs[table[s][j]]);
                }
                res.push_back(temp);
                flag[s] = true;
            }
        }
        return res;
    }
};

//最长公共子串(非子序列) http://www.lintcode.com/zh-cn/problem/longest-common-substring/

class Solution {
public:    
    /**
     * @param A, B: Two string.
     * @return: the length of the longest common substring.
     */
    int longestCommonSubstring(string &A, string &B) {
        int len1 = A.size(), len2 = B.size();
        
        int LCS = 0;
        
        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                int lcs_temp = 0;
                while (i + lcs_temp < len1 
                    && j + lcs_temp < len2 
                    && A[i + lcs_temp] == B[j + lcs_temp]) 
                        lcs_temp++;
                LCS = max(LCS, lcs_temp);
            }
        }
        return LCS;
    }
};

// Rotate Image https://leetcode.com/problems/rotate-image/

/*
上下翻转，沿对角线翻转
*/

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n; i++) {
            if (i < n - i - 1) swap(matrix[i], matrix[n - i - 1]);
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) swap(matrix[i][j], matrix[j][i]);
        }
    }
};


//Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring/

class Solution {
public:
    string longestPalindrome(string s) {
        int maxlen = 1;
        int n = s.size();
        if (n == 1) return s;
        int r[2];
        for (int i = 0; i < n - 1; i++) {
            int j = 1;
            int len = 1;
           //  string temp;
         //   temp += s[i];
            while (i - j >=0 && i + j < n) {
                if (s[i - j] == s[i + j]) {
                    len += 2;
                //     temp = s[i - j] + temp + s[i + j];
                    j++;
                }
                else break;
            }
            if (len > maxlen) {maxlen = len; r[0] = i - (j - 1); r[1] = i + (j - 1);}
            
            if (s[i] == s[i + 1]) {
                int k = 1;
                int len2 = 2;
               // string temp2;
               // temp2 += s[i];
               // temp2 += s[i + 1];
                while (i - k >= 0 && i + 1 + k < n) {
                    if (s[i - k] == s[i + 1 + k]) {
                        len2 += 2;
                      //  temp2 = s[i - k] + temp2 + s[i + 1 + k];
                        k++;
                    }
                    else break;
                }
                if (len2 > maxlen) {maxlen = len2; r[0] = i - (k -1); r[1] = i + k;}
            }
        }
        cout << r[0] << ' ' << r[1] << endl;
        return s.substr(r[0], r[1] - r[0] + 1);
    }
};

/*字序列 （可能会超时）*/
class Solution {
public:
    string longestPalindrome(string s) {
        
        return ans(0, s.size() - 1, s);
    }
    
    string ans(int i, int j, string s) {
        string temp;
        if (i > j) return temp;
        if (i == j) return temp + s[i];
        
        if (s[i] == s[j]) return s[i] + ans(i + 1, j - 1, s) + s[j];
        else {
            string s1 = ans(i + 1, j, s);
            string s2 = ans(i, j - 1, s);
            
            if (s1.size() > s2.size()) return s1;
            else return s2;
        }
    }
};

//Mini Parser https://leetcode.com/problems/mini-parser/


/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Constructor initializes an empty nested list.
 *     NestedInteger();
 *
 *     // Constructor initializes a single integer.
 *     NestedInteger(int value);
 *
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Set this NestedInteger to hold a single integer.
 *     void setInteger(int value);
 *
 *     // Set this NestedInteger to hold a nested list and adds a nested integer to it.
 *     void add(const NestedInteger &ni);
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */
class Solution {
public:
    NestedInteger deserialize(string s) {
        if (s.empty() ) return NestedInteger();
        if (s[0] != '[') return NestedInteger(stoi(s));
        if (s.size() <= 2) return NestedInteger();
        
        int start = 1, cnt = 0;
        NestedInteger res;
        for (int i = 1; i < s.size(); i++) {
            if (cnt == 0 && (s[i] == ',' || i == s.size() - 1)) {
                res.add(deserialize(s.substr(start, i - start)));
                start = i + 1;
            } else if (s[i] == '[') {
                cnt++;
            } else if(s[i] == ']') {
                cnt--;
            }
        }
        return res;
    }
};


// Flatten Nested List Iterator https://leetcode.com/problems/flatten-nested-list-iterator/

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */
class NestedIterator {
public:
    stack<NestedInteger> st;
    
    NestedIterator(vector<NestedInteger> &nestedList) {
        for (int i = nestedList.size() - 1; i >= 0; i--) {
            st.push(nestedList[i]);
        }
    }

    int next() {
        NestedInteger i = st.top();
        st.pop();
        return i.getInteger();
    }

    bool hasNext() {
        while (!st.empty() ) {
            NestedInteger t = st.top();
            if (t.isInteger() ) return true;
            st.pop();
            for (int i = t.getList().size() - 1; i >= 0; i--) {
                st.push(t.getList()[i]);
            }
        }
        return false;
    }
};

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i(nestedList);
 * while (i.hasNext()) cout << i.next();
 */


 //Palindrome Pairs https://leetcode.com/problems/palindrome-pairs/

/*
 class Solution(object):
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        wmap = {y : x for x, y in enumerate(words)}
        
        def isPalindrome(word):
            size = len(word)
            for x in range(size / 2):
                if word[x] != word[size - x - 1]:
                    return False
            return True

        ans = set()
        for idx, word in enumerate(words):
            if "" in wmap and word != "" and isPalindrome(word):
                bidx = wmap[""]
                ans.add((bidx, idx))
                ans.add((idx, bidx))

            rword = word[::-1]
            if rword in wmap:
                ridx = wmap[rword]
                if idx != ridx:
                    ans.add((idx, ridx))
                    ans.add((ridx, idx))

            for x in range(1, len(word)):
                left, right = word[:x], word[x:]
                rleft, rright = left[::-1], right[::-1]
                if isPalindrome(left) and rright in wmap:
                    ans.add((wmap[rright], idx))
                if isPalindrome(right) and rleft in wmap:
                    ans.add((idx, wmap[rleft]))
        return list(ans)

*/
        
// Integer to English Words   https://leetcode.com/problems/integer-to-english-words/

/*
class Solution(object):
    def numberToWords(self, num):
        
        if num == 0:
            return 'Zero'
            
        """
        :type num: int
        :rtype: str
        """
        units_place = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        ten_to_19 = ['Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
        tens_place = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        
        thousands_place = ['', 'Thousand', 'Million', 'Billion']
        st = []
        
        while num != 0:
            temp_string = ''
            value = num % 1000
            num = num / 1000
            
            first = value / 100
            value = value % 100
            if first != 0:
                temp_string += units_place[first - 1] + ' ' + 'Hundred' + ' '
            
            second = value / 10
            if second == 0:
                if value % 10 > 0:
                    temp_string += units_place[value % 10 - 1]
                temp_string += ' '
            elif second == 1:
                temp_string += ten_to_19[value - 10] + ' '
            else:
                temp_string += tens_place[second - 2] + ' '
                if value % 10 > 0:
                    temp_string += units_place[value % 10 - 1] + ' '
                
            st.append(temp_string)
            
        res = ''
        count = len(st)
        while len(st) != 0:
           if st[-1] != ' ':
               res += st.pop() + thousands_place[count - 1] + ' '
           else:
               st.pop()
               
           count -= 1
        
        res = ' '.join(res.split())
        return res

*/ 

//Count Primes https://leetcode.com/problems/count-primes/


class Solution {
public:
    int countPrimes(int n) {
        if (n <=  2) return 0;
        vector<bool> v(n - 1, true);
        int res = 0;
        v[0] = v[1] = false;
        int limit = sqrt(n);
        for (int i = 2; i <= limit; i++) {
            for (int j = i * i; j < n; j += i) v[j] = false;
        }
        for (int i = 0; i < n; i++) 
            if (v[i]) res++;
        return res;
    }
};


// Basic Calculator II  https://leetcode.com/problems/basic-calculator-ii/

"""
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = str(''.join(s.split()))
        
        first_op = ['+']
        first_val = re.split('[\+\-]', s)
        
        for i in s:
            if i in '+-':
                first_op.append(i)
        res = 0
        
        for i in range(len(first_val)):
            
            second_op = ['*']
            second_val = re.split(r'[\*\/]', first_val[i])
            
            for j in first_val[i]:
                if j in '/*':
                    second_op.append(j)
            
            res2 = 1
            for j in range(len(second_val)):
                if second_op[j] == '*':
                    res2 *= int(second_val[j])
                else:
                    res2 /= int(second_val[j])
                    
            first_val[i] = res2
            
            if first_op[i] == '+':
                res += first_val[i]
            else:
                res -= first_val[i]
        
        return res
                
"""
            