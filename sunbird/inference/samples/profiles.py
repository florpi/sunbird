from .base import Samples
import loggig

class Profiles(Samples):
    """
    Class to read profiles generated with sunbird's inference module.
    """
    def __init__(self, data=None):
        self.logger = logging.getLogger(__name__)
        pass