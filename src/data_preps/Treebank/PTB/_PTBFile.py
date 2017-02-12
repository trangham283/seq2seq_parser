from Treebank.Nodes import File
from _PTBNode import PTBNode
from _PTBSentence import PTBSentence
from _PTBLeaf import PTBLeaf

import os.path
from xml.etree import cElementTree as etree


class PTBFile(File, PTBNode):
    """
    A Penn Treebank file
    """
    def __init__(self, **kwargs):
        path = kwargs.pop('path')
        if 'string' in kwargs:
            text = kwargs.pop('string')
        else:
            text = open(path).read()
        # Sometimes sentences start (( instead of ( (. This is an error, correct it
        filename = path.split('/')[-1]
        self.path = path
        self.filename = filename
        self.ID = filename
        self._IDDict = {}
        PTBNode.__init__(self, label='File', **kwargs)
        if self.filename.endswith('xml'):
            root_dir = os.path.dirname(os.path.dirname(path))
            self._parseNXT(root_dir, filename.split('.')[0])
        else:
            self._parseFile(text)

    def _parseFile(self, text):
        currentSentence = []
        for line in text.strip().split('\n'):
            # Detect the start of sentences by line starting with (
            # This is messy, but it keeps bracket parsing at the sentence level
            if line.startswith('(') and currentSentence:
                self._addSentence(currentSentence)
                currentSentence = []
            currentSentence.append(line)
        self._addSentence(currentSentence)

    def _addSentence(self, lines):
        sentStr = '\n'.join(lines)[1:-1]
        nSents = len(self)+1
        sentID = '%s~%s' % (self.filename, str(nSents).zfill(4))
        self.attachChild(PTBSentence(string=sentStr, globalID=sentID, localID=self.length()))


class NXTFile(File, PTBNode):
    def __init__(self, **kwargs):
        self.path = kwargs.pop('path')
        self.filename = kwargs.pop('filename')
        self.ID = self.filename
        self._IDDict = {}
        PTBNode.__init__(self, label='File', **kwargs)
        self.xml_idx = {}
        self._parseNXT(self.path, self.filename)
        self._addTurns(self.path, self.filename)

    def _parseNXT(self, nxt_root_dir, file_id):
        terminals = {}
        ns = '{http://nite.sourceforge.net/}'
        for speaker in ['A', 'B']:
            # Get terminals
            terminals_loc = os.path.join(nxt_root_dir, 'xml', 'terminals',
                                        '%s.%s.terminals.xml' % (file_id, speaker))
            leaves_tree = etree.parse(open(terminals_loc))
            for w_xml in leaves_tree.iter('word'):
                wordID = int(w_xml.get(ns+'id').split('_')[1]) - 1
                w_node = PTBLeaf(label=w_xml.get('pos'), start_time=w_xml.get(ns+'start'),
                                 end_time=w_xml.get(ns+'end'), text=w_xml.get('orth'),
                                 wordID=wordID)
                terminals[w_xml.get(ns+'id')] = w_node
            for p_xml in leaves_tree.iter('punc'):
                wordID = int(p_xml.get(ns+'id').split('_')[1]) - 1
                terminals[p_xml.get(ns+'id')] = PTBLeaf(label=p_xml.text,
                                                     text=p_xml.text,
                                                     wordID=wordID)
            for t_xml in leaves_tree.iter('trace'):
                wordID = int(t_xml.get(ns+'id').split('_')[1]) - 1
                terminals[t_xml.get(ns+'id')] = PTBLeaf(label='-NONE-', text='-NONE-',
                                                     wordID=wordID)
            for s_xml in leaves_tree.iter('sil'):
                wordID = int(s_xml.get(ns+'id').split('_')[1]) - 1
                terminals[s_xml.get(ns+'id')] = PTBLeaf(label='-NONE-', text='-SIL-',
                                                        wordID=wordID)
                
            syntax_loc = os.path.join(nxt_root_dir, 'xml', 'syntax',
                                      '%s.%s.syntax.xml' % (file_id, speaker))
            syntax_tree = etree.parse(open(syntax_loc))
            for s_xml in syntax_tree.iter('parse'):
                localID = int(s_xml.get(ns+'id')[1:])
                globalID = '%s~%s' % (file_id, str(localID).zfill(4))
                ptb_sent = PTBSentence(xml_node=s_xml, terminals=terminals,
                                       globalID=globalID, localID=localID)
                self.xml_idx[(speaker, localID)] = ptb_sent
                self.attachChild(ptb_sent)
        self.sortChildren()

    def _addTurns(self, path, filename):
        ns = '{http://nite.sourceforge.net/}'
        for speaker in ['A', 'B']:
            turns_loc = os.path.join(path, 'xml', 'turns', '%s.%s.turns.xml' % (filename, speaker))
            turns_tree = etree.parse(open(turns_loc))
            for turn_xml in turns_tree.iter('turn'):
                child = turn_xml.getchildren()[0]
                sent_ids = child.get('href').split('#')[1]
                if '..' in sent_ids:
                    first_id, last_id = sent_ids.split('..')
                else:
                    first_id = sent_ids
                    last_id = first_id
                first_id = int(first_id[4:-1])
                last_id = int(last_id[4:-1]) + 1
                turnID = turn_xml.get(ns+'id')
                for sent_idx in range(first_id, last_id):
                    sent = self.xml_idx[(speaker, sent_idx)]
                    self.xml_idx[(speaker, sent_idx)].addTurn(speaker, turnID)
