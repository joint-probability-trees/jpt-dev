'''
Version information for jpt.
'''
import sys


__all__ = [
    'VERSION_MAJOR',
    'VERSION_MINOR',
    'VERSION_PATCH',
    'VERSION_STRING_FULL',
    'VERSION_STRING_SHORT',
    '__version__',
]


VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH = (0, 1, 11)
VERSION_STRING_SHORT = '%s.%s' % (VERSION_MAJOR, VERSION_MINOR)
VERSION_STRING_FULL = '%s.%s' % (VERSION_STRING_SHORT, VERSION_PATCH)


__version__ = VERSION_STRING_FULL


if sys.version_info[0] < 3:
    raise Exception('Unsupported Python version: %s' % sys.version_info[0])
