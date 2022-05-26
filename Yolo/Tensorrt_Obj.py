# This file Defines a tenorrt object for tensorrt inference
import os
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

import utils.engine as engine_utils # TRT Engine creation/save/load utils
import utils.model as model_utils # UFF conversion uttils

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class TRTInference(object):
    def __init__(self, engine_path,trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        # Load TRT Plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        # Initialize Runtime
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # Placehold Engine
        self.engine = None

        if os.path.exists(engine_path) and not self.engine:
            print("Loading Engine")
            self.engine = engine_utils.load_engine(self.trt_runtime, engine_path)
        else:
            print("Path not found: "+engine_path)

        # Allocate Memory for network input/output
        self.inputs, self.outputs, self.bindings, self.stream = \
        engine_utils.allocate_buffers(self.engine)

        # Set up context
        self.context = self.engine.create_execution_context()

        # Allocate Memory for Batches
        input_vol = trt.volume(model_utils.ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.engine.max_batch_size, input_vol))

    def infer(self, im):
        # Preprocess Image
        input_width = model_utils.ModelData.get_input_width()
        input_height = model_utils.ModelData.get_input_height()
        im_rsz = cv2.resize(im, (input_width, input_height))

        # HWC to CHW
        im_f = im_rsz.transpose((2,0,1))

        # Normalize
        im_f = (2.0 / 255.0) * im_f - 1
        im_f = im_f.ravel()

        # Copy to Memory
        np.copyto(self.inputs[0].host, im_f.ravel())

        #Measure inference time
        inference_start = time.time()

        # Get Output
        # Transfer input to GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        # Run Inference
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions from GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        # Synchronize Stream
        self.stream.synchronize()

        # Print time
        print("Infer Time: {} ms".format(int(round((time.time() - inference_start) * 1000))))

        # Return Results
        return [out.host for out in self.outputs]

