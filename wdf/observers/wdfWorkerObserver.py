__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"

import multiprocessing as mp
from wdf.observers.observer import Observer
from wdf.observers.observable import Observable
from wdf.processes.wdfUnitWorker import wdfUnitWorker 
from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker
import logging
from wdf.config.Parameters import Parameters

class wdfWorkerObserver(Observer, Observable):
    def __init__ ( self, parameters, fullPrint=0 , downsampling=True):
        """
        :type parameters: class Parameters object
        :param parameters: run configuration, including `nproc` (worker pool size);
            copied onto a fresh `Parameters` instance passed down to the per-segment
            worker so mutations there don't leak back into the caller's object.
        :type fullPrint: int
        :param fullPrint: verbosity passed through to the underlying worker.
        :type downsampling: bool
        :param downsampling: use `wdfUnitDSWorker` (downsampling pipeline) if True,
            else the non-downsampling `wdfUnitWorker`.
        """
        Observable.__init__(self)
        Observer.__init__(self)
        self.pool = mp.Pool(parameters.nproc)

        self.par = Parameters()
        self.par.copy(parameters)
        if downsampling:
            self.wdfworker = wdfUnitDSWorker(self.par, fullPrint)
        else:
            self.wdfworker = wdfUnitWorker(self.par, fullPrint)
                

    def wait_completion ( self ):
        """ Wait for completion of all the tasks in the queue.

        :return: None
        """
        self.pool.close()
        self.pool.join()

    def update ( self, segment, last ):
        """Dispatches `segmentProcess` call(s) to the worker pool.

        :param segment: passed straight through to `pool.map`/`pool.apply_async` as
            their respective `iterable`/`args` argument -- on the `last` branch it is
            iterated (`pool.map(self.wdfworker.segmentProcess, segment)`, one
            `segmentProcess` call per element), on the non-`last` branch it is used
            as `apply_async`'s positional-args tuple for a single call. No caller of
            this class currently exists inside wdflow itself to confirm the exact
            shape expected in each branch -- read `pool.map`/`apply_async`'s own
            argument-unpacking semantics before relying on this method.
        :type last: bool
        :param last: if True, this is the final segment -- blocks on
            `wait_completion` after dispatching instead of firing asynchronously,
            so the pool is fully drained and closed before returning.
        :return: None
        """
        try:
            if last:
                logging.info("Last job launched")
                self.pool.map(self.wdfworker.segmentProcess, segment)
                self.wait_completion()
            else:
                self.pool.apply_async(self.wdfworker.segmentProcess, segment)
        except KeyboardInterrupt:
            self.pool.terminate()
            
