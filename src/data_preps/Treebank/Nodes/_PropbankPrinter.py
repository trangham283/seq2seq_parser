from _Printer import Printer

class PropbankPrinter(Printer):
    """
    Print trees with Propbank annotation
    """
    def setEntries(self, entries):
        nodes = {}
        for entry in entries:
            for parg in entry.pargs:
                for nodeSet in parg.refChain:
                    for node in nodeSet:
                        if parg.feature:
                            label = '%s_%s' % (parg, parg.feature)
                        else:
                            label = parg.label
                        if node in nodes:
                            label = label + '-' + nodes[node]
                        nodes[node] = label
        self.entries = nodes

    def _printNode(self, node):
        """
        Print indentation, a bracket, then the node label.
        Then print the node's children, then a close bracket.
        """
        indentation = '  '*self._indentation
        if node in self.entries:
            self._lines.append('%s(%s-%s' % 
                               (indentation, node.label, self.entries[node]))
        else:
            self._lines.append('%s(%s' % (indentation, node.label))
        self._indentation += 1
        for child in node.children():
            if self._isLeaf(child):
                self._printLeaf(child)
            else:
                self._printNode(child)
        self._lines[-1] = self._lines[-1] + ')'
        self._indentation -= 1

    def _printLeaf(self, node):
        if node in self.entries:
            self._lines[-1] = self._lines[-1] + ' %s-%s' % \
               (node.text, self.entries[node])
        else:
            self._lines[-1] = self._lines[-1] + ' %s' % (node.text)
