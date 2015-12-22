from test import *
from variants import *
from pprint import pprint

variants = [
        Variant('noavx',   '-DCMAKE_CXX_FLAGS=-std=c++11 -fabi-version=6'),
        Variant('avx',     '-DCMAKE_CXX_FLAGS=-std=c++11 -mavx -fabi-version=6'),
        Variant('avx2',    '-DCMAKE_CXX_FLAGS=-std=c++11 -mavx2 -fabi-version=6'),

        Variant('release',  '-DCMAKE_BUILD_TYPE=Release'),
        Variant('debug',    '-DCMAKE_BUILD_TYPE=Debug'),

        Variant('cover',    '-DVSNRAY_ENABLE_COVER=ON'),
        Variant('examples', '-DVSNRAY_ENABLE_EXAMPLES=ON'),
        Variant('noviewer', '-DVSNRAY_ENABLE_VIEWER=OFF'),
        Variant('viewer',   'viewer', vtype='target'),

        Variant('viewerf1', '-DVIEWER_PACKET_SIZE=1'),
        Variant('viewerf4', '-DVIEWER_PACKET_SIZE=4'),
        Variant('viewerf8', '-DVIEWER_PACKET_SIZE=8'),
    ]

# convert to dict
variants = { i.name: i for i in variants }

test_list = None

test_list = specialize_variant_list(test_list, [
    'noavx',
    'avx',
    'avx2',
    ])

test_list = specialize_variant_list(test_list, [
    'release',
    'debug',
    ])

test_list = specialize_variant_list(test_list, [
    'cover+examples+noviewer',
    'viewer',
    ])

test_list = specialize_variant_list(test_list, [
    ('viewerf1', 'viewer'),
    ('viewerf4', 'viewer+avx,viewer+avx2'),
    ('viewerf8', 'viewer+avx,viewer+avx2'),
    ])

#test_list = extend_variant_list(test_list, [
#    ('viewer', 'debug'),
#    ])

pprint(test_list)

tests = variant_list_to_variant_tests(test_list, variants)

import shutil
import os.path
if os.path.exists('/media/ramfs/test_build'):
    shutil.rmtree('/media/ramfs/test_build')

for t in tests:
    print('Running {}: '.format(t.name), end='', flush=True)
    t.run_test('/media/ramfs/test_build', '..')
    print(t.get_result_string())
