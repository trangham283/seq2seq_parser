import unittest
import os.path
import os

import Treebank.PTB

class TestPTB(unittest.TestCase):
    def test_corpus(self):
        #path = '/usr/local/data/Penn3/parsed/mrg/wsj/'
        path = '/scratch/ttran/WSJ/wsj/'
        ptb = Treebank.PTB.PennTreebank(path=path)
        self.assertEqual(ptb.length(), 2312)
        asbestos = ptb.child(2).child(0)
        self.assertEqual(41, len(asbestos.listWords()))


class TestNXT(unittest.TestCase):
    def test_file(self):
        #path = '/usr/local/data/NXT-Switchboard/'
        path = '/scratch/ttran/nxt_switchboard_ann/'
        file1 = Treebank.PTB.NXTFile(path=path, filename='sw2005')
        sent = file1.child(1)
        for node in sent.depthList():
            if node.isLeaf():
                continue
            print node.duration(), node.gap_after(), ' '.join(w.text for w in node.listWords())

if __name__ == '__main__':
    unittest.main()
