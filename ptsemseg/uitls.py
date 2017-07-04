'''
Misc Utility functions
'''

import os

def recursive_glob(rootdir='.', suffix=''):
    ''' Performs recursive glob with given suffix and rootdir '''
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]