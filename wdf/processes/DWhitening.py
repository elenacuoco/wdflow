"""Whitening

.. moduleauthor:: Elena Cuoco <elena.cuoco@ego-gw.it>

"""
__author__ = "Elena Cuoco"
__project__ = "pytsa"

from pytsa.tsa import DoubleWhitening
from pytsa.tsa import SeqView_double_t as SV
from pytsa.tsa import LatticeView 

class DWhitening(object):
    
    def __init__(self, LV, OutputSize, ExtraSize):
        self.LV = LatticeView(LV)
        self.DW = DoubleWhitening(self.LV, OutputSize, ExtraSize)
        self.DW.init(self.LV) 
    
    def ParametersLoad(self, LVfile):
        """
        This method loads the calculated AR and LV parameter from the file

       

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        :return: Autoregressive and Lattice View
        """
       
        self.LV.Load(LVfile, "txt")
        ## not clear, but absolutly neeeded for initialitiate Dwhitening class
        self.LV.Load(LVfile)
       
        return   self.LV  

    def Process(self, data, dataw):  
        self.DW(data,dataw)
         
        return  
    def ProcessAsync(self, data, dataw):  
        self.DW.Input(data)
        try:
            self.DW.Output(dataw)
        except BaseException:
            print("no output data available")

        return  
    
    def Input(self,data):
        self.DW.Input(data)
        return
    
    def Output(self,data):
        try:
            self.DW.Output(data)
        except BaseException:
            print("no output data available")
        return
     
    def Init(self,LV) : 
        self.DW.init(LV) 
    
     
    def SetOutputSize(self, Nout, Extrasize):
     
        self.DW.SetOutputSize(Nout, Extrasize) 
    
    def GetDataNeeded(self):
        return self.DW.GetDataNeeded()      
     
 