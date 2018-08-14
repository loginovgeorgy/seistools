class Units(object):
    def __init__(self, distance='m', time='s', amplitude='m/s2'):
        """
        class for determining the units of measurements


        :param distance: 'm', 'km', 'ft'
        :param time: 's', 'ms',
        :param amplitude: speed, acceleration, count
        """
        self.distance = distance
        self.time = time
        self.amplitude = amplitude

    def _get_description(self):

        return 'Units: \n distance "{}" \n time "{}" \n amplitude "{}"'.format(
            self.distance,
            self.time,
            self.amplitude
        )

    def __str__(self):
        return self._get_description()

    def __repr__(self):
        return self._get_description()
