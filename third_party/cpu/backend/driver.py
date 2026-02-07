import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++", "numa"]

def get_type_size(ty):
    if ty.startswith('*'): return 8
    mapping = {"i1": 1, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "fp16": 2, "bf16": 2,
               "i32": 4, "u32": 4, "fp32": 4, "f32": 4, "i64": 8, "u64": 8, "fp64": 8}
    return mapping.get(ty, 1)

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids, algo_info=None):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    # Filter out constexpr from args_format since they're not passed at runtime
    # Note: constants keys might be tuples like (4,) instead of int 4
    def is_constant(i):
        return i in constants or (i,) in constants

    # We must strict check the filtering. If JIT passes it, we must parse it.
    # Empirically, JIT is passing constexpr args in this environment.
    # So we include everything in parsing.
    # UPDATE: When using Torch Inductor, JIT behavior might differ relative to pure Triton JIT.
    # If using inductor, constants might NOT be passed.
    # We should detect if we are running under Inductor or manual JIT?
    # Or, we can just rely on the fact that Inductor always seems to not pass constexprs.
    
    # Let's revert to filtering out constants for parsing logic, similar to how official backends might do it.
    filtered_signature_keys = [i for i in sorted(signature.keys()) if not is_constant(i)]
    
    # args_format = ''.join([format_of(signature[i]) for i in sorted(signature.keys())])
    args_format = ''.join([format_of(signature[i]) for i in filtered_signature_keys])
    
    format = "iiiOKOOOO" + args_format

    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    # Same for declarations: Declare ALL arguments that JIT passes -> NO, only runtime args
    # arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    arg_decls = ', '.join(f"{ty_to_cpp(signature[i])} arg{i}" for i in filtered_signature_keys)

    # arg_ptrs_list must match arg_decls
    # We use full signature here so ParseTuple can fill them all
    # arg_ptrs_list = ', '.join(f"&arg{i}" for i in signature.keys())
    arg_ptrs_list = ', '.join(f"&arg{i}" for i in filtered_signature_keys)
    
    # But for invalidating/using in kernel, we might still skip constants? 
    # kernel_fn_args is what we pass to the actual kernel function.
    
    # kernel_fn_args = [i for i in sorted(signature.keys()) if not is_constant(i)]
    kernel_fn_args = filtered_signature_keys
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    # The error was here: running run_omp_kernels with signature arg types, but we filtered them in arg_decls
    # run_omp_kernels definition uses arg_decls from filtered_signature_keys.
    # So the types list used for function pointer casting must also match filtered_signature_keys.
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    # Generate binding code if algo_info is provided
    binding_code = ""
    if algo_info and algo_info.get("block_size", 0) > 0 and algo_info.get("ptr_idx") is not None:
        ptr_idx = algo_info["ptr_idx"]
        blk = algo_info["block_size"]
        esz = algo_info["element_size"]
        ptr_arg = f"arg{ptr_idx}"
        # We need to ensure logic is safe. arg{ptr_idx} is a void* in C++.
        # We cast to char* for arithmetic.
        binding_code = f"""
        if (my_start < my_end) {{
             const auto [bx, by, bz] = all_grids[my_start];
             // NUMA Binding Optimization
             void* base_ptr = (void*) {ptr_arg};
             if (base_ptr) {{
                 size_t offset = (size_t)bx * {blk} * {esz};
                 void* target_addr = (char*)base_ptr + offset;
                 int node = get_memory_node(target_addr);
                 if (node >= 0) {{
                     printf("[TritonCPU] Thread bound to Node %d for addr %p\\n", node, target_addr);
                     numa_run_on_node(node);
                 }}
             }}
        }}
        """

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>
#include <numa.h>
#include <numaif.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

// 1. Function to find which NUMA node a virtual address physically resides on
int get_memory_node(void* ptr) {{
    int node = -1;
    // MPOL_F_NODE: return the node ID of the address
    // MPOL_F_ADDR: specify we are querying a specific address, not policy
    if (get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) < 0) {{
        // perror("get_mempolicy failed");
        return -1;
    }}
    return node;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} 0, 0, 0, 1, 1, 1);
      return;
  }}

  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
       int tid = 0;
       size_t my_start = 0;
       size_t my_end = N;
       {binding_code}
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

#ifdef _OPENMP
#pragma omp parallel num_threads(max_threads)
#endif // _OPENMP
  {{
     int tid = 0;
     int used_nt = 1;
#ifdef _OPENMP
     tid = omp_get_thread_num();
     used_nt = omp_get_num_threads();
#endif
     
     size_t chunk_size = N / used_nt;
     size_t remainder = N % used_nt;
     size_t my_start = tid * chunk_size + (tid < remainder ? tid : remainder);
     size_t my_end = my_start + chunk_size + (tid < remainder ? 1 : 0);

     {binding_code}

     for (size_t i = my_start; i < my_end; ++i) {{
       const auto [x, y, z] = all_grids[i];
       (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
     }}
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if signature[i][0] == "*" else "" for i in kernel_fn_args])};
  run_omp_kernels(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if signature[i][0]=="*" else f"arg{i}" for i in kernel_fn_args)) if len(kernel_fn_args) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        
        # Extract Algo Info for optimizations
        algo_info = {}
        block_size = 0
        if hasattr(src, "fn"):
            arg_names = src.fn.arg_names
            # Heuristic to find block size
            for i, name in enumerate(arg_names):
                if name in ["BLOCK_SIZE", "BLOCK_M", "BLOCK_X", "XBLOCK"]:
                    if i in constants:
                        block_size = constants[i]
                        break
                    elif (i,) in constants:
                         block_size = constants[(i,)]
                         break
        algo_info["block_size"] = block_size
        
        algo_info["ptr_idx"] = None
        algo_info["element_size"] = 1
        sorted_indices = sorted(signature.keys())
        for idx in sorted_indices:
            ty = signature[idx]
            if ty.startswith('*'):
                algo_info["ptr_idx"] = idx
                algo_info["element_size"] = get_type_size(ty[1:])
                break
        print(algo_info)
        src = make_launcher(constants, signature, ids, algo_info)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()
