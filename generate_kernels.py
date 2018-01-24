#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
import os.path
import subprocess
import shutil

# helpful for kernel development
debug = 0

gen_kernels = [
    [ "xgemm_blocksparse_32x32x32_xprop", "fprop", "A32", "B32", "C32" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "fprop", "A10", "B10", "C10" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "fprop", "A10", "B32", "C10" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "fprop", "A7",  "B7",  "C7"  ],

    [ "xgemm_blocksparse_32x32x32_xprop", "bprop", "A32", "B32", "C32" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "bprop", "A10", "B10", "C10" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "bprop", "A32", "B10", "C32" ],
    [ "xgemm_blocksparse_32x32x32_xprop", "bprop", "A7",  "B7",  "C7"  ],
    [ "xgemm_blocksparse_32x32x32_xprop", "bprop", "A32", "B7",  "C32"  ],

    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A32", "B32", "C32" ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A10", "B10", "C10" ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A10", "B32", "C10" ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A10", "B32", "C32" ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A7",  "B7",  "C7"  ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A7",  "B32", "C7"  ],
    [ "xgemm_blocksparse_32x32x8_updat",  "updat", "A7",  "B32", "C32" ],

    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F32", "I32", "O32" ],
    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F16", "I16", "O16" ],
    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F16", "I32", "O32" ],

    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F32", "I32", "O32", "overlapK" ],
    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F16", "I16", "O16", "overlapK" ],
    [ "xconv_blocksparse_32x32x16_fprop", "fprop", "F16", "I32", "O32", "overlapK" ],

    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F32", "I32", "O32" ],
    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F16", "I16", "O16" ],
    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F16", "I32", "O32" ],

    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F32", "I32", "O32", "overlapC" ],
    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F16", "I16", "O16", "overlapC" ],
    [ "xconv_blocksparse_32x32x16_bprop", "bprop", "F16", "I32", "O32", "overlapC" ],

    [ "xconv_blocksparse_32x32x32_updat", "updat", "E32", "I32", "O32" ],
    [ "xconv_blocksparse_32x32x32_updat", "updat", "E16", "I16", "O16" ],
    [ "xconv_blocksparse_32x32x32_updat", "updat", "E32", "I16", "O16" ],
    [ "xconv_blocksparse_32x32x32_updat", "updat", "E16", "I32", "O16" ],
    [ "xconv_blocksparse_32x32x32_updat", "updat", "E32", "I16", "O32" ],
    [ "xconv_blocksparse_32x32x32_updat", "updat", "E16", "I32", "O32" ],
]

kernel_specs = dict(

    xgemm_blocksparse_32x32x32_xprop=dict(basename="gemm_blocksparse_32x32x32", params="xprop_matmul", threads=128, share="(32*33)*4 +  4"),
    xgemm_blocksparse_32x32x32_updat=dict(basename="gemm_blocksparse_32x32x32", params="updat_matmul", threads=128, share="(32*32)*4 + 64"),
    xgemm_blocksparse_32x32x8_updat =dict(basename="gemm_blocksparse_32x32x8",  params="updat_matmul", threads= 32, share="(32* 8)*4 + 64"),

    xconv_blocksparse_32x32x32_fprop=dict(basename="conv_blocksparse_32x32x32", params="xprop_conv",   threads=128, share="(33+32)*32*2"  ),
    xconv_blocksparse_32x32x16_fprop=dict(basename="conv_blocksparse_32x32x16", params="xprop_conv",   threads= 64, share="(17+16)*32*2"  ),
    xconv_blocksparse_32x32x16_bprop=dict(basename="conv_blocksparse_32x32x16", params="xprop_conv",   threads= 64, share="(16+16)*32*2 + 64" ),
    xconv_blocksparse_32x32x32_updat=dict(basename="conv_blocksparse_32x32x32", params="updat_conv",   threads=128, share="32*33*4 +  4"  ),
)

_params = {
    "xprop_matmul": [
        "unsigned* param_Layout",
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float param_alpha",
        "float param_beta",
        "unsigned param_cda",
        "unsigned param_cdc",
        "unsigned param_m",
    ],
    "updat_matmul": [
        "plist8 param_A",
        "plist8 param_B",
        "unsigned* param_Layout",
        "float* param_C",
        "float param_alpha",
        "float param_beta",
        "unsigned param_cda",
        "unsigned param_cdb",
        "unsigned param_k",
        "unsigned param_count",
    ],
    "xprop_conv": [
        "unsigned* param_Block",
        "unsigned* param_LutMPQ",
        "unsigned* param_LutCK",
        "float* param_O",
        "float* param_F",
        "float* param_I",
        "float param_alpha",
        "unsigned param_TRS",
        "unsigned param_magic_TRS",
        "unsigned param_shift_TRS",
        "unsigned param_CDHW",
        "unsigned param_KMPQ",
    ],
    "updat_conv": [
        "unsigned* param_Block",
        "unsigned* param_LutMPQ",
        "unsigned* param_LutCK",
        "float* param_O",
        "float* param_E",
        "float* param_I",
        "float param_alpha",
        "unsigned param_TRS",
        "unsigned param_magic_TRS",
        "unsigned param_shift_TRS",
        "unsigned param_CDHW",
        "unsigned param_KMPQ",
        "unsigned param_N",
        "unsigned param_sizeF",
    ],
}

def _get_cache_dir(subdir=None):

    cache_dir = os.path.expanduser("~/.cache/blocksparse")

    if subdir:
        subdir = subdir if isinstance(subdir, list) else [subdir]
        cache_dir = os.path.join(cache_dir, *subdir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir

base_dir  = os.path.dirname(__file__)
maxas_dir = os.path.join(base_dir, "vendor", "maxas")
sass_dir  = os.path.join(base_dir, "src", "sass")

_space_re = re.compile(r"\s+")

_share_template = r"""
    .shared .align 4 .b32 share[{0}];
"""

_kernel_template = r"""
.version {6}
.target {0}
.address_size 64

// args: {5}

.visible .entry  {1}(
{2}
)
{{
{4}
    ret;
}}
"""
#.reqntid {3}

def get_ptx_file(kernel_spec, args_spec, kernel_name, arch, ptx_ver):

    ptx_dir = _get_cache_dir([arch, 'ptx'])

    thread_spec = kernel_spec["threads"]
    param_spec  = _params[kernel_spec["params"]]

    kernel_params = []
    for p in param_spec:
        ptype, pname = _space_re.split(p)

        if ptype == "plist8":
            kernel_params.append("    .param .align 8 .b64 %s[8]" % pname)
        else:
            if ptype[-1] == '*':
                ptype = '.u64'
            elif ptype == 'float':
                ptype = '.f32'
            else:
                ptype = '.u32'

            kernel_params.append("    .param %s %s" % (ptype, pname))

    kernel_params = ",\n".join(kernel_params)

    if "share" in kernel_spec:
        share = _share_template.format(eval(kernel_spec["share"]))
    else:
        share = ""

    kernel_text = _kernel_template.format(arch, kernel_name, kernel_params, thread_spec, share, args_spec, ptx_ver)
    kernel_ptx  = os.path.join(ptx_dir, kernel_name + ".ptx")

    current_text = ""
    if os.path.exists(kernel_ptx):
        f = open(kernel_ptx, "r")
        current_text = f.read()
        f.close()
    # only write out the kernel if text has changed.
    if kernel_text != current_text:
        f = open(kernel_ptx, "w")
        f.write(kernel_text)
        f.close()

    return kernel_ptx


include_re = re.compile(r'^<INCLUDE\s+file="([^"]+)"\s*/>')

def extract_includes(name, includes=None):
    if not includes:
        includes = list()
    sass_file = os.path.join(sass_dir, name)
    includes.append((sass_file, os.path.getmtime(sass_file)))
    for line in open(sass_file, "r"):
        match = include_re.search(line)
        if match:
            extract_includes(match.group(1), includes)
    return includes

def run_command(cmdlist):
    cmd  = " ".join(cmdlist)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("Error(%d):\n%s\n%s" % (proc.returncode, cmd, err))
    #if debug:
    print(cmd)
    if out: print(out)
    if err: print(err)

def get_kernel(kernel):

    major, minor = 5, 0
    arch = "sm_%d%d" % (major, minor)
    libprefix = "PERL5LIB=%s" % maxas_dir
    maxas_i = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -i -w"]
    maxas_p = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -p"]

    sass_name   = kernel[0]
    kernel_spec = kernel_specs[sass_name]
    kernel_name = kernel_spec["basename"]
    args_spec   = str(kernel[1:])

    for opt in kernel[1:]:
        maxas_i.append("-D%s 1" % opt)
        maxas_p.append("-D%s 1" % opt)
        kernel_name += "_" + opt

    maxas_i.insert(2, "-k " + kernel_name)

    sass_name += ".sass"
    cubin_name = kernel_name + ".cubin"
    cubin_dir  = _get_cache_dir([arch, 'cubin'])

    ptx_version = "4.2" if major < 6 else "5.0"
    ptx_file   = get_ptx_file(kernel_spec, args_spec, kernel_name, arch, ptx_version)
    sass_file  = os.path.join(sass_dir, sass_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)

    if not os.path.exists(sass_file):
        raise RuntimeError("Missing: %s for kernel: %s" % (sass_name, kernel_name))

    ptx_mtime   = os.path.getmtime(ptx_file)
    cubin_mtime = os.path.getmtime(cubin_file) if os.path.exists(cubin_file) else 0

    build_cubin = False
    if ptx_mtime > cubin_mtime:
        build_cubin = True

    includes = extract_includes(sass_name)
    for include, include_mtime in includes:
        if include_mtime > cubin_mtime:
            build_cubin = True
            break

    if build_cubin:
        # build the cubin and run maxas in the same command
        # we don't want the chance of a generated cubin not processed by maxas (in case user hits ^C in between these steps)
        run_command([ "ptxas -v -arch", arch, "-o", cubin_file, ptx_file, ";" ] + maxas_i + [sass_file, cubin_file])
        cubin_mtime = time.time()

    # output preprocessed and disassembled versions in debug mode
    if debug:
        pre_dir  = _get_cache_dir([arch, 'pre'])
        dump_dir = _get_cache_dir([arch, 'dump'])

        pre_file   = os.path.join(pre_dir,  kernel_name + "_pre.sass")
        dump_file  = os.path.join(dump_dir, kernel_name + "_dump.sass")
        pre_mtime  = os.path.getmtime(pre_file)  if os.path.exists(pre_file)  else 0
        dump_mtime = os.path.getmtime(dump_file) if os.path.exists(dump_file) else 0

        for include, include_mtime in includes:
            if include_mtime > pre_mtime:
                run_command(maxas_p + [sass_file, pre_file])
                break

        # if cubin_mtime > dump_mtime:
        #     run_command(["nvdisasm -c", cubin_file, ">", dump_file])

    return kernel_name, cubin_file


def main():
    header_file = os.path.join(base_dir, "build", "blocksparse_kernels.h")

    with open(header_file, "w") as output_file:

        kernel_map = "\n\nstd::map<std::string, std::pair<const uint8_t*, size_t>> kernel_map_ = {"

        for kernel in gen_kernels:

            kernel_name, cubin_file = get_kernel(kernel)

            kernel_text = "\n\nconst uint8_t %s[] = {" % kernel_name
            with open(cubin_file, 'rb') as input_file:
                count = 0
                byte = input_file.read(1)
                use_hex = 'hex' in dir(byte)
                while byte:
                    if count % 32 == 0:
                        kernel_text += "\n    "
                    count += 1
                    if use_hex:
                        kernel_text += "0x" + byte.hex() + ","
                    else:
                        kernel_text += "0x" + byte.encode("hex") + ","
                    byte = input_file.read(1)
                kernel_text += "\n};"
                kernel_map += "\n    { \"%s\", { %s, %d } }," % (kernel_name, kernel_name, count)


            output_file.write(kernel_text)

        kernel_map += "\n};"
        output_file.write(kernel_map)

if __name__ == '__main__':
    main()
