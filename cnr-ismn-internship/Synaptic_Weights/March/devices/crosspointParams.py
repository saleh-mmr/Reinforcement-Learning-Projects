class CrossPointParams:
    def __init__(self, g_ap_coefficient, g_p_coefficient, shift_parameter, g_bias_coefficient, noise_stddev, ):
        self.g_ap_coefficient = g_ap_coefficient
        self.g_p_coefficient = g_p_coefficient
        self.shift_parameter = shift_parameter
        self.g_bias_coefficient = g_bias_coefficient
        self.noise_stddev = noise_stddev

    def get_noise_stddev(self):
        return self.noise_stddev