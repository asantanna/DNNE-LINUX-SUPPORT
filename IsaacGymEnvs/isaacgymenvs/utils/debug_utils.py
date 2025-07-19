# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Debug utilities that don't depend on torch or other heavy imports"""

import inspect


# Debug print function for consistent logging
def DNNE_print(shared, category, message):
    """Print with [DNNE_DEBUG] shared/category format for easy grep filtering
    
    Args:
        shared: "D" for DNNE only, "I" for IGE only, "B" for both (shared code)
        category: Category string (e.g., "PPO_CYCLE", "PPO_BATCH", "ENV_INIT")
        message: The message to print
    """
    print(f"[DNNE_DEBUG] {shared}/{category}: {message}")


def debug_get_calling_function_name(levels_up=2):
    """Get the name of the calling function for debug purposes
    
    Args:
        levels_up: How many levels up the call stack to look (default 2)
                  1 = immediate caller of this function
                  2 = caller of the function that called this (default)
                  
    Returns:
        String with format "module.Class.method" or "module.function"
    """
    try:
        frame = inspect.currentframe()
        # Go up the specified number of levels
        for _ in range(levels_up):
            if frame is not None:
                frame = frame.f_back
            else:
                return "unknown"
        
        if frame is None:
            return "unknown"
            
        # Get the function/method name
        func_name = frame.f_code.co_name
        
        # Try to get the class name if it's a method
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__
            module_name = frame.f_locals['self'].__class__.__module__
            # Get just the last part of the module name
            module_short = module_name.split('.')[-1] if module_name else ""
            return f"{module_short}.{class_name}.{func_name}"
        else:
            # It's a function, not a method
            module_name = frame.f_globals.get('__name__', '')
            module_short = module_name.split('.')[-1] if module_name else ""
            return f"{module_short}.{func_name}"
            
    except Exception as e:
        # Don't let debug code crash the program
        return f"error: {str(e)}"
    finally:
        # Clean up frame reference to avoid memory leaks
        if 'frame' in locals():
            del frame