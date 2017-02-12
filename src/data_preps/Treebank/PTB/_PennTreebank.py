from Treebank.Nodes import Corpus
from _PTBNode import PTBNode
from _PTBFile import PTBFile
from _PTBFile import NXTFile

import os
import sys
from os.path import join as pjoin

class PennTreebank(PTBNode, Corpus):
    """
    The Penn Treebank, specifically the WSJ
    Children are built just-in-time
    """
    fileClass = PTBFile
    def __init__(self, path=None, **kwargs):
        self.path = path
        PTBNode.__init__(self, label='Corpus', **kwargs)
        for fileLoc in self._getFileList(self.path):
            self.attachChild(fileLoc)
                            
    def section(self, sec):
        for i, fileLoc in enumerate(self._children):
            path, fileName = os.path.split(fileLoc)
            if int(fileName[4:6]) == sec:
                yield self.child(i)

    def section00(self):
        for i in xrange(99):
            yield self.child(i)

    def twoTo21(self):
        for i in xrange(199, 2074):
            yield self.child(i)

    def section23(self):
        for i in xrange(2157, 2257):
            yield self.child(i)

    def section24(self):
        for i in xrange(2257, self.length()):
            yield self.child(i)

    def _getFileList(self, location):
        """
        Get all files below location
        """
        paths = []
        for path in [pjoin(location, f) for f in os.listdir(location)]:
            if path.endswith('CVS'):
                continue
            elif path.startswith('.'):
                continue
            if os.path.isdir(path):
                paths.extend(self._getFileList(path))
            elif path.endswith('.mrg') or path.endswith('.auto'):
                paths.append(path)
        paths.sort()
        return paths


class NXTSwitchboard(PTBNode, Corpus):
    """The Nite XML-toolkite formatted Switchboard spoken language treebank"""
    fileClass = NXTFile
    def __init__(self, path=None, **kwargs):
        self.path = path
        PTBNode.__init__(self, label='Corpus', **kwargs)
        for filename in self._getFileList(self.path):
            self.attachChild(filename)

    def attachChild(self, filename):
        self._children.append(filename)
            
    def child(self, index):
        """
        Read a file by zero-index offset
        """
        filename = self._children[index]
        print >> sys.stderr, filename
        return self.fileClass(path=self.path, filename=filename)
 
    def _getFileList(self, location):
        location = pjoin(location, 'xml', 'syntax')
        files = set() 
        for path in [f for f in os.listdir(location)]:
            if path.startswith('.'):
                continue
            elif path.startswith('CVS'):
                continue
            assert path.endswith('xml'), path
            filename = path.split('/')[-1]
            fileID = filename.split('.')[0]
            files.add(fileID)
        files = list(files)
        files.sort()
        return files

    def train_files(self):
        for i, f in enumerate(self._children):
            if f.startswith('sw2') or f.startswith('sw3'):
                yield self.child(i)

    def dev_files(self):
        for i, f in enumerate(self._children):
            filenum = int(f[2:])
            #if 4500 < filenum <= 4936:
            if 4518 < filenum <= 4936:
                yield self.child(i)
 
    def dev2_files(self):
        for i, f in enumerate(self._children):
            filenum = int(f[2:])
            #if 4154 < filenum < 4500:
            if 4153 < filenum <= 4483:
                yield self.child(i)

    def eval_files(self):
        for i, f in enumerate(self._children):
            filenum = int(f[2:])
            #if 4000 < filenum <= 4154:
            # actually test = 4004-4153
            if 4003 < filenum <= 4153:
                yield self.child(i)

