__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
__status__ = "Development"
__project__ = "wdf"

import json
import copy
import pprint 
pp = pprint.PrettyPrinter(indent=4)

# class to handle the parameters
class Parameters(object):
    """
    This class stores set of parameters required by WDF to work
    """

    def __init__(self, **kwargs):
        """
        The constructor
        """

        self.__dict__ = kwargs

        def __getattr__(self, attr):
            return self.__dict__[attr]

    def dump(self, filename):
        """

        :param filename: name of file where saving the parameters
        :type filename: basestring
        """
        self.filename = filename
        with open(self.filename, mode="w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)

    def print(self):
        
        """
        print parameters
        """  
        
        pp.pprint(self.__dict__)   

    def load(self, filename):
        """

                :param filename: name of file where loading the parameters
                :type filename: basestring
                """
        self.filename = filename
        with open(self.filename) as data_file:
            data = json.load(data_file)
        self.__dict__ = data
        return self.__dict__

    def copy(self, param):
        """
                :param param: parameters

              """
        self.__dict__ = copy.deepcopy(param.__dict__)
        return self.__dict__
