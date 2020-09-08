# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:25:39 2016

@author: aitor
"""

#USAGE: python3 action2vec.py kasteren_dataset/actions.txt /results/actions_w1.model /results/actions_w1.vector
 
import logging
import os.path
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print((globals()['__doc__'] % locals()))
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    # avg activity lenght = 3.69590167482
    model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=0, iter=100,
            workers=multiprocessing.cpu_count())
 
    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    print('FIN')  