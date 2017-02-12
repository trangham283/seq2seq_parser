"""
Corpus construction functions
"""
import Nodes
import PTB
import os

def makeCorpus(location):
    """
    Instantiate a Corpus class
    """
    corpus = PTB.PennTreebank(path=location)
    return corpus
    #    sentenceConstructor = SentenceConstructor()
    #SWBDConstructor     = SWBDSentenceConstructor()
    #fileConstructor     = FileConstructor()
    #fileConstructor.addChildConstructors({'standard': sentenceConstructor, 'swbd': SWBDConstructor})
    #corpusConstructor   = CorpusConstructor()
    #corpusConstructor.addChildConstructors({'standard': fileConstructor})
    #return corpusConstructor.make('standard', location, Nodes.Corpus)
    
def makeDBCorpus(section):
    sentenceConstructor = SentenceConstructor
    SWBDConstructor     = SWBDSentenceConstructor
    fileConstructor     = DBFileConstructor, {'standard': sentenceConstructor, 'swbd': SWBDConstructor}
    corpusConstructor   = DBCorpusConstructor, {'database': fileConstructor}
    return corpusConstructor.make('database', section, Nodes.Corpus)
    
def makeXMLCorpus(location):
    fileConstructor     = XMLFileConstructor
    corpusConstructor   = CorpusConstructor, {'standard': fileConstructor}
    fileConstructor.setPath(location)
    return corpusConstructor.make('standard', location, Nodes.Corpus)

def fileList(location):
    """
    Get all files below location
    """
    paths = []
    for path in [os.path.join(location, fileName) for fileName in os.listdir(location)]:
        if path.endswith('CVS') or path.endswith('.SVN'): continue
        if os.path.isdir(path):
            paths.extend(fileList(path))
        elif path.endswith('.mrg') or path.endswith('.auto'):
            paths.append(path)
    paths.sort()
    return paths

def profile(function):
    import hotshot, hotshot.stats
    prof = hotshot.Profile('/tmp/test.prof')
    prof.runcall(function)
    prof.close()
    stats = hotshot.stats.load('/tmp/test.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(20)

def loadFiles():
    #location = '/home/matt/code/repos/data/TreeBank3/parsed/mrg/wsj/'
    location = '/scratch/ttran/WSJ/wsj/'
    corpus = makeCorpus(location)
    for child in corpus.children():
        pass

if __name__ == '__main__':
    profile(loadFiles)
