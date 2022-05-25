import numpy as np


from mmdaw.abstract import DriftDetector

from mmdaw.bucket_stream import BucketStream
from mmdaw.mmd import MMD


class MMDAWAdapter(DriftDetector):
    def __init__(self, gamma, alpha=0.1):
        """
        :param gamma: The scale of the data
        :param alpha: alpha value for the hypothesis test

        """
        self.gamma = gamma
        self.alpha = alpha
        self.logger = None
        self.detector = BucketStream(gamma=self.gamma, compress=True, alpha=self.alpha)
        self.element_count = 0
        super(MMDAWAdapter, self).__init__()

    def name(self) -> str:
        return "MMDAW"

    def parameter_str(self) -> str:
        return r"$\alpha = {}$".format(self.alpha)

    def pre_train(self, data):
        self.gamma = MMD.estimate_gamma(data)
        self.detector = BucketStream(gamma=self.gamma, compress=True, alpha=self.alpha)

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.element_count += 1
        self.detected_cp = False
        prev_cps = len(self.detector.get_changepoints())
        self.detector.insert(input_value[0])
        if len(self.detector.get_changepoints()) > prev_cps:
            self.delay = self.element_count - self.detector.get_changepoints()[-1]
            self.detected_cp = True

    def detected_change(self):
        return self.detected_cp

    def metric(self):
        return 0
