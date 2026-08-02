__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
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
        :type self.parameters: class self.parameters object
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
        """ Wait for completion of all the tasks in the queue """
        self.pool.close()
        self.pool.join()

    def update ( self, segment, last ):
        try:
            if last:
                logging.info("Last job launched")
                self.pool.map(self.wdfworker.segmentProcess, segment)
                self.wait_completion()
            else:
                self.pool.apply_async(self.wdfworker.segmentProcess, segment)
        except KeyboardInterrupt:
            self.pool.terminate()
            
