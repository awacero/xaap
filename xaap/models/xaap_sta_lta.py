
class StaLta:

    def __init__(self,**kwargs):

        self.kwargs = kwargs
        ##TODO : CHOSE NAME OR STATION AS ID
        self.name = kwargs.get("name")
        self.sta = kwargs.get("sta")
        self.lta = kwargs.get("lta")
        self.trigon = kwargs.get("trigon")
        self.trigoff = kwargs.get("trigoff")

