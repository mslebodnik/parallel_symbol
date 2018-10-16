import logging 
import pika
import cPickle as pickle
import zlib
import base64
import threading 
from tpot import TPOTRegressor


FORMAT = '%(asctime)s -%(threadName)11s-\t[%(levelname)s]:%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger()
IP = "czcholsint896.prg-dc.dhl.com"

def decode_data(data):
    st = zlib.decompress(base64.b64decode(data))
    return pickle.loads(st)

def encode_data(data):
    p = pickle.dumps(data)
    z = zlib.compress(p)
    return base64.b64encode(z)


class Consumer(threading.Thread):
    def __init__(self):
        self._event = threading.Event()
        self._wait = 20
        self.lock = threading.Lock()
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=IP))
        self.msg_data = self.connection.channel()

        self.msg_data.queue_declare(queue='tpot_data')
        self.msg_data.queue_declare(queue='tpot_pipelines')
        self.msg_data.basic_qos(prefetch_count=1)
        super(Consumer, self).__init__()

    def run(self):
        """
        Send heartbeat to servers, in order to prevent close connection
        """
        while not self._event.isSet():
            with self.lock:
                self.connection.process_data_events()
            logger.info("heartbeat sent")
            self._event.wait(self._wait)

    def stop(self):
        self._event.set()
        self.join()
        self.msg_data.stop_consuming()

    def consume(self):
        self.msg_data.basic_consume(self.callback, 'tpot_data')
        self.msg_data.start_consuming()

    def callback(self, channel, method, properties, body):
        with self.lock:
            (symbol, X_train, X_test, y_train, y_test, folds_index) = decode_data(body)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("data received %s %d", symbol, folds_index)
        tpot = TPOTRegressor(memory='auto', generations=100, population_size=100, n_jobs=-1,max_time_mins=20, max_eval_time_mins=20,config_dict='TPOT light')
        try:
            tpot.fit(X_train, y_train)
        except Exception as e:
            logger.error(e)
            data = (None, None, None, None)
            with self.lock:
                channel.basic_publish(exchange='', routing_key='tpot_pipelines',body=encode_data(data))
            return
           
        test_prediction = tpot.predict(X_test)
        test_prediction_error = abs((y_test-test_prediction)*100/y_test)
        score = tpot.score(X_test, y_test)
        logger.info("sending result of %s %s", symbol, folds_index)
        try:
            data = (tpot.fitted_pipeline_, score, folds_index, symbol)
            with self.lock:
                channel.basic_publish(exchange='', routing_key='tpot_pipelines',body=encode_data(data))
        except Exception:
            import pdb
            pdb.set_trace()

logger.info("Start")
c  = Consumer()
try:
    c.start()
    c.consume()
except KeyboardInterrupt:
    c.stop()
