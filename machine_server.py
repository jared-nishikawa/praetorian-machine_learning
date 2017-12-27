#!/usr/bin/python

import requests
import logging
import base64
import time
import tools
import classify
import numpy as np
from keras.models import load_model

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r

    def win(self):
        r = self._request("/hash", method="post", \
                data={"email": "jared@secureset.com"})
        return r

if __name__ == "__main__":
    # create the server object
    s = Server()

    # load our keras model
    model = load_model('model.h5')

    # Read in train.csv for corpus
    C = tools.read_data('train.csv')
    qty = 100
    C.set_common_words(qty=qty)
    C.set_vector_mapping(qty=qty)

    answers = {}
    total = 0
    correct = 0
    for _ in range(10000):
        # query the /challenge endpoint
        s.get()

        # Make a document
        binary = s.binary
        arch = ""
        D = tools.Document(arch, binary, C)
        sample = D.clean()
        batch = np.array([sample])
        Z = model.predict(batch, batch_size=1, verbose=1)
        arch, conf = classify.guess(Z[0])

        target = arch

        s.post(target)

        #s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(target, s.ans, s.wins))
        total += 1
        if target == s.ans:
            correct += 1

        print "Predicted:", target
        print "True:", s.ans

        print "%d/%d" % (correct, total)

        # 500 consecutive correct answers are required to win
        # very very unlikely with current code
        if s.hash:
            s.log.info("You win! {}".format(s.hash))
            with open('win.txt','w') as f:
                f.write("You win! {}".format(s.hash))
            print s.win()
