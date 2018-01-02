from xml.dom.minidom import parseString

XML = """
<nodeA>
    <nodeB>Text hello</nodeB>
    <nodeC><noText></noText></nodeC>
</nodeA>
"""


def replaceText(node, newText):
    if node.firstChild.nodeType != node.TEXT_NODE:
        raise Exception("node does not contain text")

    node.firstChild.replaceWholeText(newText)

def main():
    doc = parseString(XML)

    node = doc.getElementsByTagName('nodeB')[0]
    replaceText(node, "Hello World")

    print(doc.toxml())

    try:
        node = doc.getElementsByTagName('nodeC')[0]
        replaceText(node, "Hello World")
    except:
        print("error")


if __name__ == '__main__':
    main()