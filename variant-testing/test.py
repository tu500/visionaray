#!/usr/bin/python3

import os
import os.path
import re
import subprocess
import time


class VariantTest():

    def __init__(self, name, folder, cmake_flags, make_target=None):

        self.name = name
        self.folder = folder
        self.cmake_flags = cmake_flags
        self.make_target = make_target

        if self.make_target is None:
            self.make_target = 'all'

    def run_test(self, build_base_dir, project_base_dir):
        """
        build_base_dir: root dir of all test builds
        project_base_dir: root dir of tested project (visionaray source tree)
        """

        self.build_base_dir = build_base_dir
        self.project_base_dir = project_base_dir

        # create working dir
        # this will be the root dir of this test's cmake build
        self.working_dir = os.path.join(build_base_dir, self.folder)
        os.makedirs(self.working_dir, exist_ok=True)

        self.test_output = b''

        if not self._call_cmake():
            return False

        if not self._call_make():
            return False

        self.test_passed = True
        return True

    def _call_cmake(self):

        start_time = time.time()
        child = subprocess.Popen(
                ['cmake'] + self.cmake_flags + [os.path.abspath(self.project_base_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.working_dir,
            )

        returncode = child.wait()
        self.cmake_time = time.time() - start_time
        self.test_output += child.stdout.read()

        if returncode != 0:
            self.test_passed = False
            return False

        return True

    def _call_make(self):

        start_time = time.time()
        child = subprocess.Popen(
                ['make', '-j4', self.make_target],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.working_dir,
            )

        returncode = child.wait()
        self.make_time = time.time() - start_time
        self.test_output += child.stdout.read()

        if returncode != 0:
            self.test_passed = False
            return False

        return True

    def get_result_string(self):

        if self.test_passed:
            return 'Passed in {} / {} seconds'.format(self.cmake_time, self.make_time)
        else:
            return 'Failed'


class FPSTest(VariantTest):

    #TODO these two should be customizable
    test_file_path = os.path.abspath('../../teapot.obj')
    binary_rel_path = os.path.join('src', 'fpstest', 'fpstest')

    def __init__(self, name, folder, cmake_flags, make_target):
        super(FPSTest, self).__init__(name, folder, cmake_flags, make_target)

    def run_test(self, build_base_dir, project_base_dir):

        # will set self.test_passed, but we will reset it
        if not super(FPSTest, self).run_test(build_base_dir, project_base_dir):
            return False

        if not self._run_fpstest():
            return False

        self.test_passed = True
        return True

    def _run_fpstest(self):

        binary_path = os.path.abspath(os.path.join(self.working_dir, self.binary_rel_path))

        child = subprocess.Popen(
                [binary_path, self.test_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.working_dir,
            )

        returncode = child.wait()
        child_output = child.stdout.read()
        self.test_output += child_output

        match = re.match(r'FPS: (\d+\.\d+)', child_output.decode('utf-8'))

        if returncode != 0 or not match:
            self.test_passed = False
            return False

        self.fps_result = float(match.group(1))

        return True

    def get_result_string(self):

        if self.test_passed:
            return '{} FPS'.format(self.fps_result)
        else:
            return 'Failed'



def variant_line_to_variant_test(vline, defined_variants):

    name = '+'.join(vline)
    folder = os.path.join(*vline)
    flags = []
    target = None

    for item in vline:

        try:
            variant = defined_variants[item]
        except KeyError:
            raise Exception('Variant with name {} not defined'.format(repr(item)))

        if variant.vtype == 'cl-option':
            flags.append(variant.value)

        elif variant.vtype == 'target':

            if target is not None:
                raise Exception('Multiple targets configured in vline {}'.format(repr(vline)))

            target = variant.value

        elif variant.vtype == 'fpstest':

            # TODO this is ugly
            do_fpstest = True

        else:
            raise Exception('Unknown value type {} in variant'.format(repr(variant.vtype)))

    if do_fpstest:
        return FPSTest(name, folder, flags, target)

    return VariantTest(name, folder, flags, target)

def variant_list_to_variant_tests(vlist, defined_variants):
    return [variant_line_to_variant_test(l, defined_variants) for l in vlist]
