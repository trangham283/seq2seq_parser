from _PTBNode import PTBNode

class TracedNode(PTBNode):
    """
    A node referred to by a trace, so that other elements can pretend
    it's actually the original

    TODO: Test!!!
    """
    def __init__(self, **kwargs):
        """
        Store the target and the trace's constituent
        """
        self.trace = kwargs.pop('trace')
        self.target = kwargs.pop('target')
        self.traceType = trace.text.split('-')[0]
        self._parent = None
        self.globalID = trace.globalID
        self.label = trace.parent().label
        self.functionLabels = trace.parent().functionLabels

    def getWordID(self, worNum):
        return self.trace.wordID

    def children(self):
        return [self.target]
