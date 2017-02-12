import nltk.corpus

from _Node import Node
from _Printer import Printer


class Sentence(Node):
    _printer = Printer()
    def __str__(self):
        return self._printer(self)
        
    def parent(self):
        """
        Raises an error, because the root node has no parent
        """
        raise AttributeError, "Cannot retrieve the parent of the root node! Current parse state:\n\n%s" % self.prettyPrint()
        
    def performOperation(self, operation):
        """
        Accept a Visitor and call it on each child
        Goofy name/design is legacy from when I didn't know how to code :(
        """
        operation.newStructure()
        operation.actOn(self)
        for node in self.depthList():
            try:
                operation.actOn(node)
            # Give operations the opportunity to signal
            # when the work is complete
            except Break:
                break
        while operation.moreChanges:
            operation.actOn(self)
            for node in getattr(self, operation.listType)():
                try:
                    operation.actOn(node)
                # Give operations the opportunity to signal
                # when the work is complete
                except Break:
                    break

    def isRoot(self):
        return True
    
    def addSenses(self):
        for word in self.listWords():
            if word.isTrace():
                continue
            elif word.isPunct():
                word.synsets.append('punct')
                word.supersenses.append('punct')
            if word.label.startswith('V'):
                pos = nltk.corpus.wordnet.VERB
            elif word.label.startswith('N'):
                pos = nltk.corpus.wordnet.NOUN
            else:
                pos = None
            lemma = nltk.corpus.wordnet.morphy(word.text, pos=None)
            if not lemma:
                lemma = word.text
            word.addSenses(lemma, nltk.corpus.wordnet.synsets(lemma, pos))

    def _connectNodes(self, nodes, parentage):
        # Build the tree
        offsets = sorted(nodes.keys())
        # Skip the top node
        #offsets.pop(0)
        for key in offsets:
            if key not in parentage:
                continue
            node = nodes[key]
            try:
                parent = nodes[parentage[key]]
            except:
                print 'hi'
                print key
                print node
                print parentage
                print nodes
                raise
            parent.attachChild(node, len(parent))
