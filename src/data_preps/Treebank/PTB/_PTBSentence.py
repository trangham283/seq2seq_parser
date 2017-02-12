import re

from Treebank.Nodes import Sentence
from _PTBNode import PTBNode
from _PTBLeaf import PTBLeaf

class PTBSentence(PTBNode, Sentence):
    """
    The root of the parse tree
    
    Has no parent, and one or more children
    """
    def __init__(self, **kwargs):
        if 'string' in kwargs:
            node = self._parseString(kwargs.pop('string'))
        elif 'node' in kwargs:
            node = kwargs.pop('node')
        elif 'xml_node' in kwargs:
            node = self._parseNXT(kwargs.pop('xml_node'), kwargs.pop('terminals'))
        globalID = kwargs.pop('globalID')
        localID = kwargs.pop('localID')
        self.speaker = None
        self.turnID = None
        PTBNode.__init__(self, label='S', **kwargs)
        self.globalID = globalID
        self.localID = localID
        self.attachChild(node)

    bracketsRE = re.compile(r'(\()([^\s\)\(]+)|([^\s\)\(]+)?(\))')
    def _parseString(self, sent_text):
        openBrackets = []
        parentage = {}
        nodes = {}
        nWords = 0
        
        # Get the nodes and record their parents
        for match in self.bracketsRE.finditer(sent_text):
            open_, label, text, close = match.groups()
            if open_:
                assert not close
                assert label
                openBrackets.append((label, match.start()))
            else:
                assert close
                label, start = openBrackets.pop()
                if text:
                    newNode = PTBLeaf(label=label, text=text, wordID=nWords)
                    nWords += 1
                else:
                    newNode = PTBNode(string=label)
                if openBrackets:
                    # Store parent start position
                    parentStart = openBrackets[-1][1]
                    parentage[start] = parentStart
                else:
                    top = newNode
                # Organise nodes by start
                nodes[start] = newNode
        try:
            self._connectNodes(nodes, parentage)
        except:
            print sent_text
            raise
        by_identifier = dict((n.identifier, n) for n in top.depthList() if n.identifier)
        for node in top.depthList():
            if node.identified:
                node.traced = by_identifier.get(node.identified)
        return top
        
    def _parseNXT(self, root, terminals):
        nodes = {}
        parentage = {}
        ns = '{http://nite.sourceforge.net/}'
        parse_id = root.get(ns+'id')
        for xml_node in root.iter('nt'):
            id_ = xml_node.get(ns+'id')
            node = PTBNode(label=xml_node.get('cat'), start_time=xml_node.get(ns+'start'),
                           end_time=xml_node.get(ns+'end'))
            nodes[id_] = node
            for child in xml_node.getchildren():
                if child.tag != 'nt':
                    continue
                child_id = child.get(ns+'id')
                parentage[child_id] = id_
            for leaf_link in xml_node.getchildren():
                if leaf_link.tag != ns+'child':
                    continue
                xml_id = leaf_link.get('href').split('id')[1][1:-1]
                node = terminals.get(xml_id)
                if node is not None and node.text != '-SIL-':
                    nodes[xml_id] = node
                    parentage[xml_id] = id_
        self._connectNodes(nodes, parentage)
        for xml_node in root.getchildren():
            if xml_node.tag == 'nt':
                top = nodes[xml_node.get(ns+'id')]
                break
        top.sortChildren()
        for node in top.depthList():
            node.sortChildren()
        return top

    def addTurn(self, speaker, turnID):
        self.speaker = speaker
        self.turnID = turnID
