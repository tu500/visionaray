from test import *
from variants import *
from pprint import pprint
import os.path

variants = [
        Variant('noavx',    '-DCMAKE_CXX_FLAGS=-std=c++11 -fabi-version=6'),
        Variant('avx',      '-DCMAKE_CXX_FLAGS=-std=c++11 -mavx -fabi-version=6'),
        Variant('avx2',     '-DCMAKE_CXX_FLAGS=-std=c++11 -mavx2 -fabi-version=6'),

        Variant('release',  '-DCMAKE_BUILD_TYPE=Release'),
        Variant('debug',    '-DCMAKE_BUILD_TYPE=Debug'),

        Variant('cover',    '-DVSNRAY_ENABLE_COVER=ON'),
        Variant('examples', '-DVSNRAY_ENABLE_EXAMPLES=ON'),
        Variant('noviewer', '-DVSNRAY_ENABLE_VIEWER=OFF'),
        Variant('viewer',   make_targets='viewer'),

        Variant('viewerf1', '-DVIEWER_PACKET_SIZE=1'),
        Variant('viewerf4', '-DVIEWER_PACKET_SIZE=4'),
        Variant('viewerf8', '-DVIEWER_PACKET_SIZE=8'),

        Variant('fps', '-DVSNRAY_ENABLE_FPSTEST=ON', fpstest_binary=os.path.join('src','fpstest','fpstest')),

        Variant('algo_simple',       '-DFPSTEST_ALGO=ALGO_SIMPLE'),
        Variant('algo_whitted',      '-DFPSTEST_ALGO=ALGO_WHITTED'),
        Variant('algo_pathtracing',  '-DFPSTEST_ALGO=ALGO_PATHTRACING'),

        Variant('sched_tiled',       '-DFPSTEST_SCHEDULER=SCHED_TILED'),
        Variant('sched_simple',      '-DFPSTEST_SCHEDULER=SCHED_SIMPLE'),
        Variant('sched_tbb',         '-DFPSTEST_SCHEDULER=SCHED_TBB'),
    ]

# convert to dict
variants = { i.name: i for i in variants }

flag_list = None
flag_list = specialize_variant_list(flag_list, [
    'noavx',
    'avx',
    'avx2',
    ])

flag_list = specialize_variant_list(flag_list, [
    'release',
    #'debug',
    ])

test_list = flag_list
test_list = specialize_variant_list(test_list, [
    'cover+examples+noviewer',
    'viewer',
    ])

test_list = specialize_variant_list(test_list, [
    ('viewerf1', 'viewer'),
    ('viewerf4', 'viewer+avx,viewer+avx2'),
    ('viewerf8', 'viewer+avx,viewer+avx2'),
    ])

fps_list = flag_list
fps_list = specialize_variant_list(fps_list, [
    'algo_simple',
    'algo_whitted',
    'algo_pathtracing',
    ])

fps_list = specialize_variant_list(fps_list, [
    'sched_tiled',
    'sched_simple',
    'sched_tbb',
    ])

fps_list = specialize_variant_list(fps_list, [
    'fps+noviewer',
    ])

#test_list = extend_variant_list(test_list, [
#    ('viewer', 'debug'),
#    ])

pprint(test_list + fps_list)

#tests = variant_list_to_variant_tests(test_list, variants)
tests = variant_list_to_variant_tests(fps_list, variants)

import shutil
import os.path
if os.path.exists('/media/ramfs/test_build'):
    shutil.rmtree('/media/ramfs/test_build')

for t in tests:
    print('Running {}: '.format(t.name), end='', flush=True)
    t.run_test('/media/ramfs/test_build', '..')
    print(t.get_result_string())
