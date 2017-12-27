#!/usr/bin/python

import requests
import logging
import base64
import time
import random

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

    answers = {}
    for _ in range(50):
        # query the /challenge endpoint
        s.get()
        b64 = base64.b64encode(s.binary)

        # choose a random target
        target = random.choice(s.targets)
        s.post(target)

        answers[b64] = s.ans


    # test.csv or train.csv
    with open('demo.csv','w') as f:
        f.write('binary,architecture\n')
        for key in answers.keys():
            f.write(','.join([key, answers[key]]) + '\n')
