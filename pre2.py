import os, re
import pydot

def loadDotFile(dotfileDirPath: str):
    for parent, dirs, files in os.walk(dotfileDirPath):
        for file in files:
            file = os.path.join(parent, file)
            ty = os.path.splitext(file)[1]
            if ty == ".dot":
                try:
                    yield file
                except Exception as e:
                    print(e)


def tokenGen(node):
    replace_dict = {
        "\"" : " ",
        "," : " , ",
        ":" : " ",
        "(" : " ( ",
        ")" : " ) "
    }

    for key in replace_dict:
        node = node.replace(key, replace_dict[key])
    after_replace = node.lower()
    sentences = []
    for word in after_replace.split():
        if re.match("^@.*(good|bad)", word):
            sentences.append("@func")
        else:
            sentences.append(word)
    return sentences


def output(tokenlist):
    f = open(r"/home/chenzx/CPG_token/token.txt", "a")
    for token in tokenlist:
        f.write(token+"\n")
    f.close()


def main(path):
    for file in loadDotFile(path):
        try:
            dot = pydot.graph_from_dot_file(file)
            nodes = dot[0].get_nodes()
            if nodes is not None:
                for node in nodes:
                    node = node.obj_dict['attributes']['label']
                    tokenlist = tokenGen(node)
                    output(tokenlist)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    path = r"/home/chenzx/CPG_set/badCPG"
    main(path)