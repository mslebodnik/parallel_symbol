import datetime
import pandas as pd
import requests_cache
import logging
import cPickle as pickle
import zlib
import base64
import pika
import numpy as np
import sklearn.utils as sku
import pandas_datareader as pdr
from sklearn.model_selection import TimeSeriesSplit
from threading import Thread, Lock


FORMAT = '%(asctime)s -%(threadName)11s-\t[%(levelname)s]:%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
logger = logging.getLogger()
logger.info("Start")
 
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=datetime.timedelta(days=5))
symbols_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
symbols = list(symbols_table.loc[:, "Ticker symbol"])[400:]
logger.info("Symbols ready")
 
 
unprocessed_symbols = []  
data_raw = {}
for symbols_index, symbol in enumerate(symbols):
    try:
        data_raw[symbol] = pdr.quandl.QuandlReader(symbol, start=datetime.datetime(2010, 1, 1), end=datetime.datetime(2015, 1, 1), session=session, api_key='M7Hd_VxTyUK7nDfMnF5J').read().iloc[::-1] # result is time-reversed            
    except Exception as e:
        unprocessed_symbols.append(symbol)
        continue
    if data_raw[symbol].empty:
        unprocessed_symbols.append(symbol)
        continue
       
logger.info("Data downloaded")
 
data = {}
for symbols_index, symbol in enumerate(symbols):
    if symbol in unprocessed_symbols:
        continue
   
    data_raw_norm = data_raw[symbol].divide(data_raw[symbol].iloc[0])-1    
    for column in data_raw_norm.columns:
        if (pd.notnull(data_raw_norm[[column]])&np.isfinite(data_raw_norm[[column]])).all()[0] == False:
            data_raw_norm[[column]] = data_raw[symbol][[column]]
    data[symbol] = sku.Bunch(data=data_raw_norm[:-1], target=np.ravel(data_raw_norm[['AdjClose']])[1:])

 
logger.info("Symbols normalized")

def decode_data(data):
    st = zlib.decompress(base64.b64decode(data))
    return pickle.loads(st)

def encode_data(data):
    p = pickle.dumps(data)
    z = zlib.compress(p)
    return base64.b64encode(z)

class Master(Thread):
    def __init__(self):
        self.lock = Lock()
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host='czcholsint896.prg-dc.dhl.com'))
        self._data = self._connection.channel()
        self._data.queue_declare(queue='tpot_data')
        self._data.queue_declare(queue='tpot_pipelines')
        self.to_process = 0
        self.done = False
        super(Master, self).__init__()
    
    def callback(self, ch, method, properties, body):
        """
        callback for queue processing
        """
        with self.lock:
            (raw_pipeline, score, folds_index, symbol) = decode_data(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.to_process -= 1
        
        logging.info(" data received for %s %s", symbol, folds_index)

        if symbol is not None:
           with open('pipelines/{}_ppl_{}.dump'.format(symbol, folds_index), "wb") as f_ppl, open('pipelines/{}_ppl_{}_score.dump'.format(symbol, folds_index), "wb") as f_score:
                pickle.dump(raw_pipeline, f_ppl)
                pickle.dump(score,f_score)

        if self.done:
            # lock should not be necessary here, but just to be 
            # consistent
            with self.lock:
                if self.to_process == 0:
                    self._data.stop_consuming()
                    #local_con.close()

    def run(self):
        """
        Main thread method
        """
        self._data.basic_consume(self.callback, 'tpot_pipelines')
        logging.info("start consumer")
        self._data.start_consuming()

    def feed_consumers(self, data, symbols, unprocessed_symbols):
        """
        send data to queue
        """
        logger.info("Sending ready")
        tscv = TimeSeriesSplit(n_splits=10)
        for symbols_index, symbol in enumerate(symbols):
            if symbol in unprocessed_symbols:
                continue
               
            folds_index = 1
            for train_index, test_index in tscv.split(data[symbol].data):
                X_train, X_test = data[symbol].data.values[train_index], data[symbol].data.values[test_index]
                y_train, y_test = data[symbol].target[train_index], data[symbol].target[test_index]
                logger.info("processing %s %d", symbol, folds_index)
                mq_body = encode_data((symbol, X_train, X_test, y_train, y_test, folds_index))
                with self.lock:
                    self._data.basic_publish(exchange='', routing_key='tpot_data',body=mq_body)
                    self.to_process += 1
                folds_index += 1
        self.done = True

    def keep_alive(self):
        with self.lock:
            self._connection.process_data_events()
        
m = Master()
m.feed_consumers(data, symbols, unprocessed_symbols)
# start_consuming is not thread safe
m.start()

logger.info("all data send to workers waiting to finish")
while True:
    m.join(10)
    if m.isAlive():
        m.keep_alive()
    else:
        break
