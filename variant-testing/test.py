#!/usr/bin/python3

import os
import os.path
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

        # create working dir
        working_dir = os.path.join(build_base_dir, self.folder)
        os.makedirs(working_dir, exist_ok=True)

        self.test_output = b''

        # call cmake
        start_time = time.time()
        child = subprocess.Popen(
                ['cmake'] + self.cmake_flags + [os.path.abspath(project_base_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=working_dir,
            )

        returncode = child.wait()
        self.cmake_time = time.time() - start_time
        self.test_output += child.stdout.read()

        if returncode != 0:
            self.test_passed = False
            return False

        # call make
        start_time = time.time()
        child = subprocess.Popen(
                ['make', '-j4', self.make_target],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=working_dir,
            )

        returncode = child.wait()
        self.make_time = time.time() - start_time
        self.test_output += child.stdout.read()

        if returncode != 0:
            self.test_passed = False
            return False

        self.test_passed = True
        return True


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

        else:
            raise Exception('Unknown value type {} in variant'.format(repr(variant.vtype)))

    return VariantTest(name, folder, flags, target)

def variant_list_to_variant_tests(vlist, defined_variants):
    return [variant_line_to_variant_test(l, defined_variants) for l in vlist]
