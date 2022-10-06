class XaapFilter:
    
    #def __init__(self, name, type,frequency_1, frecuency_2, order=4, zerophase=False):
    def __init__(self,**kwargs):   
        """Construct a filter object
            NOTE: The default parameters values for order and zerophase are 4 and False in OBSPY 
        """
        self.kwargs = kwargs
        
        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self.frequency_1 = kwargs.get('frequency_1')
        self.frequency_2 = kwargs.get('frequency_2', None)
        self.order = kwargs.get('order',None)
        self.zerophase = kwargs.get('zerophase',None)
        """
        self.name = name
        self.type = type
        self.frequency_1 = frequency_1 
        self.frequency_2 = frecuency_2
        self.order = order 
        self.zerophase = zerophase
        """
        