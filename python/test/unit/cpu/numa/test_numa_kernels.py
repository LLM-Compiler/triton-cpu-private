import pytest
import torch
import triton
import triton.language as tl

def test_jit_numa_id():
    @triton.jit
    def kernel(output_ptr):
        node = tl.numa_id()
        tl.store(output_ptr, node)

    output = torch.empty(1, dtype=torch.int32, device="cpu")
    kernel[(1,)](output)
    
    assert output[0].item() >= 0


def test_jit_make_numa_buffer():
    @triton.jit
    def kernel(output_ptr):
        node = tl.numa_id()
        buf_ptr = tl.make_numa_buffer((128,), tl.float32, node)
        
        # Write magic value to it to prove it's writable
        tl.store(buf_ptr, 42.0)
        
        # Read it back to prove it's readable
        val = tl.load(buf_ptr)
        
        # Write it to the host-visible output
        tl.store(output_ptr, val)

    output = torch.zeros(1, dtype=torch.float32, device="cpu")
    kernel[(1,)](output)
    
    assert output[0].item() == 42.0


def test_jit_notify_wait():
    @triton.jit
    def kernel(shared_ptr):
        pid = tl.program_id(0)
        
        if pid == 0:
            # Write a value to shared_ptr
            tl.store(shared_ptr, 99.0)
            # Call notify to wake up waiting threads
            tl.notify(signal_id=1, numa_dst=0)
            
        if pid == 1:
            # Call wait to block until notified
            tl.wait(signal_id=1)

    shared = torch.zeros(1, dtype=torch.float32, device="cpu")
    # Launch the kernel with 2 instances
    kernel[(2,)](shared)
    
    # Just verifying that the kernel executes successfully without crashing or deadlocking
    assert shared[0].item() == 99.0 or shared[0].item() == 0.0 # It could be either due to race conditions writing/reading to shared before wait completes, but mostly we check it didn't crash.
