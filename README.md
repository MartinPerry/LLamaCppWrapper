Simple LLama.cpp wrapper in C++.

Decision models (https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp) are re-implemented 
from llama.cpp `tools/server/server-decision.cpp` with Codex and manual interventions.

The code does not use the common library; everything is dependent only on `llama.h` (or `mtmd.h` for multimodality). In some cases, `llama-ext.h` is also used.

Official llama.cpp does not have decision models usable directly from the C API. I did not want to call the server API via HTTP requests in my project, so I have used Codex
to extract the code. 

Note:
main.cpp contains examples. They have hardcoded paths to models, but it's just an overview of how to use the code, so it should be ok :-)
Visual Studio project contains a hard-coded path to llama.cpp build and include directory.
