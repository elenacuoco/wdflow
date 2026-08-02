__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
__status__ = "Development"


class Observable(object):
    """
    This class registers and updates various observers
    """

    def __init__(self):
        """
        This class registers and updates various observers
        """
        self.observers = []

    def register(self, observer):
        """
        This method registers an observer

        :type observer: object
        :param observer: An observer to be registered
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        """
        This methods unregisters an observer

        :type observer: object
        :param observer: An observer to be unregistered
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        """
        This method unregisters all observers
        """
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        """
        This method calls an update function for each observers with various parameters

        :type args: object
        :param args: First parameter of the update function for the given observer

        :type kwargs: object
        :param kwargs: The following parameters of the update function for the given observer

        :return: The object with triggers; type of object depends on the observer
        """
        for observer in self.observers:
            observer.update(*args, **kwargs)
