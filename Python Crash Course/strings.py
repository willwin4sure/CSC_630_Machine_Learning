import numpy as np

# str.replace()
def replace(str, sub, repl):
    output = ''
    i = 0
    while i < len(str):
        if (str[i:i+len(sub)] == sub):
            output += repl
            i += len(sub)
        else:
            output += str[i]
            i += 1

    return output

# print(replace("hello my name is the bob and i the am running out of the time the", "the", "crap"))

# str.count()
def count(str, sub):
    count = 0
    for i in range(len(str)-len(sub)):
        if (str[i:i+len(sub)] == sub):
            count += 1
    return count

# print(count("l;knmao;eijgo;aiemg;aioej;oiajemf", ";"))


def longest_word(str):
    # return str.split(' ')[np.argmax(list(map(len, str.split(' '))))]
    return max(str.split(' '), key=len)

# print(longest_word("there are some longer words in this sentence than other words hi williams"))

def longest(str):
    # best_start = 0
    # best_end = 0
    # start = 0
    # end = 0
    # for i in range(len(str)):
    #     if (str[i] == ' ' or i == len(str)-1):
    #         if (end - start) > (best_end - best_start):
    #             best_start = start
    #             best_end = end
    #         start, end = i+1, i+1
    #     else:
    #         end += 1

    # return best_start, best_end
    start = str.index(longest_word(str))
    end = start + len(longest_word(str))
    return start, end

string = "there are some longer words in this sentence than other words hi williams longestadfasdf"
splice = longest(string)
print(splice)
print(string[splice[0]:splice[1]])

def is_balanced(str):
    count = 0
    for i in range(len(str)):
        if (str[i] == '('):
            count += 1
        elif (str[i] == ')'):
            count -= 1
        
        if (count < 0):
            return False

    if (count == 0):
        return True

    return False

# print(is_balanced("(8x + (46^3)) = 24"))
# print(is_balanced(")()("))
# print(is_balanced("(()(()()))"))
# print(is_balanced("(()(()())))"))
# print(is_balanced("(()(()())(())))"))

[int]