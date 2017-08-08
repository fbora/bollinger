# references:
# http://qpython.readthedocs.io/en/latest/usage-examples.html#subscribing-to-tick-service

import numpy as np
import pandas as pd
import threading
import sys

from qpython import qconnection
from qpython.qtype import QException
from qpython.qconnection import MessageType
from qpython.qcollection import QTable


class ListenerThread(threading.Thread):

    def __init__(self, q):
        super(ListenerThread, self).__init__()
        self.q = q
        self._stopper = threading.Event()
        self.midnight = pd.to_datetime(pd.datetime.today().strftime('%Y-%m-%d'))

    def stopit(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()

    def run(self):
        while not self.stopped():
            print('.')
            try:
                message = self.q.receive(data_only = False, raw = False) # retrieve entire message

                if message.type != MessageType.ASYNC:
                    print('Unexpected message, expected message of type: ASYNC')

                if isinstance(message.data, list):
                    # unpack upd message
                    if len(message.data) == 3 and message.data[0] == b'upd':
                        for row in message.data[2]:
                            timestamp = self.midnight + message.data[2].timestamp.values[0]
                            ticker = message.data[2].sym.values[0].decode('utf-8')
                            price = message.data[2].price.values[0]
                            print(timestamp, ticker, price)

            except QException as e:
                print(e)


if __name__ == '__main__':
    with qconnection.QConnection(host = 'localhost', port = 5017) as q:
        print(q)
        print('IPC version: %s. Is connected: %s' % (q.protocol_version, q.is_connected()))
        print('Press <ENTER> to close application')

        # subscribe to tick
        response = q.sync('.u.sub', numpy.string_('trade'), numpy.string_(['FB', 'GOOG']))
        # get table model
        if isinstance(response[1], QTable):
            print('%s table data model: %s' % (response[0], response[1].dtype))

        t = ListenerThread(q)
        t.start()
        sys.stdin.readline()
        t.stopit()
