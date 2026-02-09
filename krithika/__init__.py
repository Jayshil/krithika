from .version import version as __version__

# Then you can be explicit to control what ends up in the namespace,
__all__ = ['krithika', 'ndimageviewer', 'utils']

from .krithika import *
from .ndimageviewer import NDImageViewer
from .utils import *