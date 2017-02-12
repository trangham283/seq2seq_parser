from Treebank.Nodes import Leaf
from _PTBNode import PTBNode

class PTBLeaf(Leaf, PTBNode):
    def __init__(self, **kwargs):
        self.wordID = kwargs.pop('wordID')
        self.text = kwargs.pop('text')
        if self.text.startswith('*ICH*'):
            kwargs['identified'] = self.text.split('-')[1]
        self.synsets = []
        self.supersenses = []
        self.lemma = self.text
        if kwargs['label'].startswith('^'):
            kwargs['label'] = kwargs['label'][1:]
        kwargs['label'] = kwargs['label'].split('^')[0]
        PTBNode.__init__(self, **kwargs)

    def isEdited(self):
        node = self.parent()
        while not node.isRoot():
            if node.label == 'EDITED':
                return True
            node = node.parent()
        return False

    def isPartial(self):
        if not self.isPunct() and self.text.endswith('-') or self.label == 'XX':
            return True
        else:
            return False
