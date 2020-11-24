import re

def preprocessing(string: str):
    replace_dict = {
        "\"" : " ",
        "," : " , ",
        ":" : " ",
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