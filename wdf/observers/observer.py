__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = ["http://www.giantflyingsaucer.com/"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
__status__ = "Development"

from abc import ABCMeta, abstractmethod


class Observer(object):
    """
    The observer class
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        pass
