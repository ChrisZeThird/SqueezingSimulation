class Crystal(object):
    def __init__(self, crystal='KTP', pump_polarization="V", signal_polarization="V", idler_polarization="V"):
        self.crystal = crystal
        self.pump_polarization = pump_polarization
        self.signal_polarization = signal_polarization
        self.idler_polarization = idler_polarization

    # def Sellmeier(self):
    #     if self.crystal.upper() in ['KTP', 'LN']: