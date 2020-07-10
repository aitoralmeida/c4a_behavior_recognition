# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:49:41 2016
@author: aitor
"""

# USAGE: python train_word2vec_model.py wiki.en.text wiki.en.text.model wiki.en.text.vector

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
        print()
        globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    # avg activity lenght = 3.69590167482
    model = Word2Vec(LineSentence(inp), size=250, window=2, min_count=3,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
    print('FIN')