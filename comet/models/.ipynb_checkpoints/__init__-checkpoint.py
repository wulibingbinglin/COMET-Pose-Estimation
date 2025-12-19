# 在 Python 项目中，__init__.py 文件通常用于将某些模块引入到包中，使得其他模块可以通过包的名称直接引用。
# 比如在 vggsfm.models 文件夹下有一个 __init__.py 文件，那么在导入模块时，vggsfm.models 会被认为是一个包，
# 并且可以通过 from vggsfm.models import VGGSfM 来导入。
# models 是一个包，包中通常包含多个模块和类。在 Python 中，包只是一个文件夹，其中有一个 __init__.py 文件，它允许我们把这个文件夹当作一个包来引用。
# 但变成包之后，里面的所有类只要在下面进行声明，就可以直接用vggsfm.models.类名进行使用，例如：vggsfm.models.VGGSfM

from .E2Epose2 import VGGSFM

from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.base_track_predictor import BaseTrackerPredictor

from .track_predictor import TrackerPredictor
from .camera_predictor9 import CameraPredictor
# from .triangulator import Triangulator

__all__ = [
    'VGGSFM',  # 确保类名在这里
    'BaseTrackerPredictor',
    'BasicEncoder',
    'ShallowEncoder',
    'CameraPredictor',
    'TrackerPredictor',
]

