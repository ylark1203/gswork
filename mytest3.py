import sys
import nvdiffrast.torch as dr

print("before"); sys.stdout.flush()
glctx = dr.RasterizeGLContext()
print("after"); sys.stdout.flush()
