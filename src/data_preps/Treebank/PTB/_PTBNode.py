import re
import bisect

from Treebank.Nodes import Node


class PTBNode(Node):
    """
    A node in a parse tree
    """
    _labelRE = re.compile(r'([^-=]+)(?:-([^-=\d]+))?(-UNF)?(?:-(\d+))?(?:=(\d+))?')
    def __init__(self, **kwargs):
        def parse_time(time_str):
            if time_str is None or time_str == 'n/a':
                return None
            elif time_str == 'non-aligned' or time_str == '?':
                return -1
            else:
                return float(time_str)

        string = kwargs.pop('string', None)
        if string is not None:        
            pieces = PTBNode._labelRE.match(string).groups()
            label, functionLabel, unf, identifier, identified = pieces
            start_time = None
            end_time = None
        else:
            label = kwargs.pop('label')
            functionLabel = kwargs.pop('functionLabel', None)
            identifier = kwargs.pop('identifier', None)
            identified = kwargs.pop('identified', None)
            unf = kwargs.pop('unf', False)
            start_time = parse_time(kwargs.pop('start_time', None))
            end_time = parse_time(kwargs.pop('end_time', None))
        if kwargs:
            raise StandardError, kwargs
        if functionLabel == 'UNF':
            functionLabel = None
            unf = True
        self.start_time = start_time
        self.end_time = end_time
        self.functionLabel = functionLabel
        self.identifier = identifier
        self.identified = identified
        self.unf = bool(unf)
        # This will be filled by a node that a trace indicates
        self.traced = None
        Node.__init__(self, label)

    def duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        if self.start_time == -1 or self.end_time == -1:
            return None
        return self.end_time - self.start_time

    def gapAfter(self):
        if self.end_time < 0:
            return None
        words = self.root().listWords()
        idx = words.index(self.getWord(-1))
        if idx == len(words) - 1:
            return None
        else:
            next_word = words[idx + 1]
        if next_word.start_time < 0:
            return None
        return next_word.start_time - self.end_time
        return words[wordID].start_time - self.end_time

    def head(self):
        """
        This should really implement the Magerman head-finding
        heuristics, but currently selects last word
        """
        # TODO Implement Magerman head finding
        return self.getWord(-1)

