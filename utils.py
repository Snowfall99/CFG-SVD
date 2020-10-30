import re

def preprocessing(string: str):
    '''
    完成替换，分词，返回list

    Args:
        string:str dot文件中node节点label，llvm指令

    Return：
        list[str]
    '''
    replace_dict = {"\l": " ",
                    "(": " ( ",
                    ")": " ) ",
                    "[": " [ ",
                    "]": " ] ",
                    "{": " { ",
                    "}": " } ",
                    "\"": " ",
                    ",": " , ",
                    ":": " ",
                    "align 1": "align1",
                    "align 2": "align2",
                    "align 4": "align4",
                    "align 8": "align8",
                    "align 16": "align16",
                    }
    for key in replace_dict:
        string = string.replace(key, replace_dict[key])
    after_replace = string.lower()
    sentences = []
    for word in after_replace.split():
        if re.match("^@.*(good|bad)", word):
            sentences.append("@func")
        else:
            sentences.append(word)
    return sentences